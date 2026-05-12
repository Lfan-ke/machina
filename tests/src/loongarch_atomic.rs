use machina_accel::code_buffer::CodeBuffer;
use machina_accel::ir::tb::{EXCP_LOONGARCH_DONE, EXCP_UNDEF, TB_EXIT_IDX1};
use machina_accel::ir::Context;
use machina_accel::translate::translate_and_execute;
use machina_accel::{HostCodeGen, X86_64CodeGen};
use machina_guest_loongarch::loongarch::cpu::{
    LoongArchCpu, GUEST_BASE_CPU_OFFSET,
};
use machina_guest_loongarch::loongarch::csr::{
    CRMD_DA, CRMD_PG, CSR_BADV, CSR_CRMD, CSR_ERA, CSR_ESTAT, CSR_PRMD,
    CSR_TLBEHI, CSR_TLBELO0, CSR_TLBELO1, CSR_TLBIDX, CSR_TLBRERA,
    CSR_TLBRPRMD,
};
use machina_guest_loongarch::loongarch::exception::ECODE_PME;
use machina_guest_loongarch::loongarch::ext::LoongArchCfg;
use machina_guest_loongarch::loongarch::trans::{
    LoongArchDisasContext, LoongArchTranslator,
};
use machina_guest_loongarch::translator_loop;

const OP_ADD_D: u32 = 0b00000000000100001;
const OP_ADDI_D: u32 = 0b0000001011;
const OP_LD_D: u32 = 0b0010100011;
const OP_LD_WU: u32 = 0b0010101010;
const OP_ST_W: u32 = 0b0010100110;
const OP_LL_W: u32 = 0b00100000;
const OP_SC_W: u32 = 0b00100001;
const OP_LL_D: u32 = 0b00100010;
const OP_SC_D: u32 = 0b00100011;
const OP_AMSWAP_D: u32 = 0b00111000011000001;
const OP_AMADD_W: u32 = 0b00111000011000010;
const OP_AMADD_D: u32 = 0b00111000011000011;
const OP_AMSWAP_DB_D: u32 = 0b00111000011010011;
const OP_AMADD_DB_W: u32 = 0b00111000011010100;
const OP_AMMAX_WU: u32 = 0b00111000011001110;
const OP_AMMIN_DU: u32 = 0b00111000011010001;
const OP_RDTIMEL_W: u32 = 0b0000000000000000011000;
const OP_RDTIMEH_W: u32 = 0b0000000000000000011001;
const OP_RDTIME_D: u32 = 0b0000000000000000011010;
const OP_REVH_2W: u32 = 0b0000000000000000010000;
const OP_REVH_D: u32 = 0b0000000000000000010001;
const OP_DBAR: u32 = 0b00111000011100100;
const OP_IBAR: u32 = 0b00111000011100101;
const OP_BEQZ: u32 = 0b010000;
const OP_BNEZ: u32 = 0b010001;
const ERTN_INSN: u32 = 0x0648_3800;

fn r3(op: u32, rk: u32, rj: u32, rd: u32) -> u32 {
    (op << 15) | (rk << 10) | (rj << 5) | rd
}

fn r2(op: u32, rj: u32, rd: u32) -> u32 {
    (op << 10) | (rj << 5) | rd
}

fn r2_si12(op: u32, si12: i16, rj: u32, rd: u32) -> u32 {
    (op << 22) | ((si12 as u16 as u32 & 0x0FFF) << 10) | (rj << 5) | rd
}

fn r2_si14(op: u32, si14: i16, rj: u32, rd: u32) -> u32 {
    (op << 24) | ((si14 as u16 as u32 & 0x3FFF) << 10) | (rj << 5) | rd
}

fn r1_offs21(op: u32, offs21: i32, rj: u32) -> u32 {
    let imm = offs21 as u32 & 0x001F_FFFF;
    (op << 26)
        | (((imm >> 16) & 0x1F) << 0)
        | ((imm & 0xFFFF) << 10)
        | (rj << 5)
}

fn code15(op: u32, code: u32) -> u32 {
    (op << 15) | (code & 0x7FFF)
}

fn run_la(cpu: &mut LoongArchCpu, insns: &[u32]) -> usize {
    let code: Vec<u8> =
        insns.iter().flat_map(|insn| insn.to_le_bytes()).collect();

    let mut backend = X86_64CodeGen::new();
    backend.set_guest_base_offset(GUEST_BASE_CPU_OFFSET);
    let mut buf = CodeBuffer::new(4096).unwrap();
    backend.emit_prologue(&mut buf);
    backend.emit_epilogue(&mut buf);

    let mut ir = Context::new();
    backend.init_context(&mut ir);

    let mut ctx =
        LoongArchDisasContext::new(0, code.as_ptr(), LoongArchCfg::default());
    ctx.base.max_insns = insns.len() as u32;
    translator_loop::<LoongArchTranslator>(&mut ctx, &mut ir);

    unsafe { translate_and_execute(&mut ir, &backend, &mut buf, cpu.env_ptr()) }
}

fn read_u32(mem: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(mem[off..off + 4].try_into().unwrap())
}

fn read_u64(mem: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(mem[off..off + 8].try_into().unwrap())
}

fn tlbelo_test(ppn: u64, v: bool, d: bool, plv: u8) -> u64 {
    u64::from(v)
        | (u64::from(d) << 1)
        | (u64::from(plv & 0x3) << 2)
        | ((ppn & 0xF_FFFF_FFFF) << 12)
        | (1 << 6)
}

fn write_test_tlb_entry(
    cpu: &mut LoongArchCpu,
    index: usize,
    va: u64,
    page_size: u8,
    elo0: u64,
    elo1: u64,
) {
    let pair_mask = !((1_u64 << (u32::from(page_size) + 1)) - 1);
    cpu.csr_write(CSR_TLBEHI, (va & pair_mask) & !0x1FFF);
    cpu.csr_write(CSR_TLBELO0, elo0);
    cpu.csr_write(CSR_TLBELO1, elo1);
    cpu.csr_write(CSR_TLBIDX, (u64::from(page_size) << 24) | index as u64);
    cpu.tlb_write(index);
}

#[test]
fn loongarch_ll_sc_pairs_update_memory_and_status() {
    let mut mem = [0u8; 64];
    mem[8..12].copy_from_slice(&0xFFFF_FFFEu32.to_le_bytes());
    mem[16..24].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.set_ram_base(0);
    cpu.set_ram_end(mem.len() as u64);
    cpu.write_gpr(2, 0);

    let insns = [
        r2_si14(OP_LL_W, 2, 2, 5),
        r3(OP_ADD_D, 0, 5, 6),
        r2_si12(OP_ADDI_D, 0x123, 0, 5),
        r2_si14(OP_SC_W, 2, 2, 5),
        r2_si14(OP_LL_D, 4, 2, 7),
        r3(OP_ADD_D, 0, 7, 8),
        r2_si12(OP_ADDI_D, 0x456, 0, 7),
        r2_si14(OP_SC_D, 4, 2, 7),
    ];

    assert_eq!(run_la(&mut cpu, &insns), 0);
    assert_eq!(cpu.read_gpr(6), (-2i64) as u64);
    assert_eq!(cpu.read_gpr(5), 1);
    assert_eq!(read_u32(&mem, 8), 0x123);
    assert_eq!(cpu.read_gpr(8), 0x1122_3344_5566_7788);
    assert_eq!(cpu.read_gpr(7), 1);
    assert_eq!(read_u64(&mem, 16), 0x456);
}

#[test]
fn loongarch_sc_fails_without_or_after_lost_reservation() {
    let mut mem = [0u8; 64];
    mem[8..12].copy_from_slice(&0xAAAA_AAAAu32.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.set_ram_base(0);
    cpu.set_ram_end(mem.len() as u64);
    cpu.write_gpr(2, 0);
    cpu.write_gpr(5, 0x1111_1111);
    cpu.write_gpr(9, 0xBBBB_BBBB);

    let insns = [
        r2_si14(OP_SC_W, 2, 2, 5),
        r2_si14(OP_LL_W, 2, 2, 6),
        r2_si12(OP_ST_W, 8, 2, 9),
        r2_si12(OP_ADDI_D, 0x333, 0, 6),
        r2_si14(OP_SC_W, 2, 2, 6),
    ];

    assert_eq!(run_la(&mut cpu, &insns), 0);
    assert_eq!(cpu.read_gpr(5), 0);
    assert_eq!(cpu.read_gpr(6), 0);
    assert_eq!(read_u32(&mem, 8), 0xBBBB_BBBB);
}

#[test]
fn loongarch_sc_status_is_visible_to_following_branch() {
    let mut mem = [0u8; 64];
    mem[8..16].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.set_ram_base(0);
    cpu.set_ram_end(mem.len() as u64);
    cpu.write_gpr(2, 0);

    let success_insns = [
        r2_si14(OP_LL_D, 2, 2, 5),
        r2_si12(OP_ADDI_D, 0, 0, 5),
        r2_si14(OP_SC_D, 2, 2, 5),
        r1_offs21(OP_BEQZ, 1, 5),
    ];

    assert_eq!(run_la(&mut cpu, &success_insns), TB_EXIT_IDX1 as usize);
    assert_eq!(cpu.read_gpr(5), 1);
    assert_eq!(read_u64(&mem, 8), 0);

    mem[8..12].copy_from_slice(&0xAAAA_AAAAu32.to_le_bytes());
    cpu.write_gpr(5, 0);
    cpu.write_gpr(8, 0);

    let failure_insns = [
        r2_si14(OP_LL_W, 2, 2, 5),
        r2_si12(OP_ADDI_D, 0x33, 0, 8),
        r2_si12(OP_ST_W, 8, 2, 8),
        r2_si12(OP_ADDI_D, 7, 0, 5),
        r2_si14(OP_SC_W, 2, 2, 5),
        r1_offs21(OP_BNEZ, 1, 5),
    ];

    assert_eq!(run_la(&mut cpu, &failure_insns), TB_EXIT_IDX1 as usize);
    assert_eq!(cpu.read_gpr(5), 0);
    assert_eq!(read_u32(&mem, 8), 0x33);
}

#[test]
fn task83_store_conditional_dirty_fault_traps_instead_of_failing() {
    let cases = [
        ("sc.w", 4usize, OP_LL_W, OP_SC_W, 0x1122_3344_u64, 0x55_u64),
        (
            "sc.d",
            8usize,
            OP_LL_D,
            OP_SC_D,
            0x1122_3344_5566_7788_u64,
            0x66_u64,
        ),
    ];

    for (name, size, ll_op, sc_op, initial, replacement) in cases {
        let va = 0x4000_u64;
        let mut mem = [0u8; 64];
        mem[..size].copy_from_slice(&initial.to_le_bytes()[..size]);

        let mut cpu = LoongArchCpu::new();
        cpu.set_guest_base(mem.as_mut_ptr() as u64);
        cpu.set_ram_base(0);
        cpu.set_ram_end(mem.len() as u64);
        cpu.csr_write(CSR_CRMD, CRMD_PG);
        cpu.write_gpr(2, va);

        write_test_tlb_entry(
            &mut cpu,
            machina_guest_loongarch::loongarch::mmu::mtlb_flat_index(0)
                .unwrap(),
            va,
            12,
            tlbelo_test(0, true, false, 0),
            tlbelo_test(1, true, true, 0),
        );

        let exit = run_la(
            &mut cpu,
            &[
                r2_si14(ll_op, 0, 2, 5),
                r2_si12(OP_ADDI_D, replacement as i16, 0, 5),
                r2_si14(sc_op, 0, 2, 5),
            ],
        );

        assert_eq!(exit, EXCP_LOONGARCH_DONE as usize, "{name}");
        assert_eq!(cpu.read_gpr(5), replacement, "{name}");
        assert_eq!(
            &mem[..size],
            &initial.to_le_bytes()[..size],
            "{name} must not write dirty page"
        );
        assert_eq!(cpu.csr_read(CSR_BADV), va, "{name}");
        assert_eq!(
            (cpu.csr_read(CSR_ESTAT) >> 16) & 0x3F,
            u64::from(ECODE_PME),
            "{name}"
        );
    }
}

#[test]
fn loongarch_atomic_rmw_variants_return_old_and_update_memory() {
    let mut mem = [0u8; 64];
    mem[0..4].copy_from_slice(&0xFFFF_FFFEu32.to_le_bytes());
    mem[8..16].copy_from_slice(&10u64.to_le_bytes());
    mem[16..20].copy_from_slice(&0x8000_0001u32.to_le_bytes());
    mem[24..32].copy_from_slice(&u64::MAX.wrapping_sub(5).to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.write_gpr(2, 0);
    cpu.write_gpr(3, 8);
    cpu.write_gpr(4, 16);
    cpu.write_gpr(5, 24);
    cpu.write_gpr(11, 5);
    cpu.write_gpr(13, 0x1122_3344_5566_7788);
    cpu.write_gpr(15, 2);
    cpu.write_gpr(17, 7);

    let insns = [
        r3(OP_AMADD_W, 11, 2, 10),
        r2_si12(OP_LD_WU, 0, 2, 18),
        r3(OP_AMSWAP_D, 13, 3, 12),
        r2_si12(OP_LD_D, 0, 3, 19),
        r3(OP_AMMAX_WU, 15, 4, 14),
        r2_si12(OP_LD_WU, 0, 4, 20),
        r3(OP_AMMIN_DU, 17, 5, 16),
        r2_si12(OP_LD_D, 0, 5, 21),
    ];

    assert_eq!(run_la(&mut cpu, &insns), 0);
    assert_eq!(cpu.read_gpr(10), (-2i64) as u64);
    assert_eq!(cpu.read_gpr(18), 3);
    assert_eq!(cpu.read_gpr(12), 10);
    assert_eq!(cpu.read_gpr(19), 0x1122_3344_5566_7788);
    assert_eq!(cpu.read_gpr(14), 0xFFFF_FFFF_8000_0001);
    assert_eq!(cpu.read_gpr(20), 0x8000_0001);
    assert_eq!(cpu.read_gpr(16), u64::MAX.wrapping_sub(5));
    assert_eq!(cpu.read_gpr(21), 7);
}

#[test]
fn task47_atomic_db_variants_return_old_and_update_memory() {
    let mut mem = [0u8; 32];
    mem[0..4].copy_from_slice(&10u32.to_le_bytes());
    mem[8..16].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.write_gpr(2, 0);
    cpu.write_gpr(3, 8);
    cpu.write_gpr(11, 5);
    cpu.write_gpr(13, 0x8877_6655_4433_2211);

    let insns = [r3(OP_AMADD_DB_W, 11, 2, 10), r3(OP_AMSWAP_DB_D, 13, 3, 12)];

    assert_eq!(run_la(&mut cpu, &insns), 0);
    assert_eq!(cpu.read_gpr(10), 10);
    assert_eq!(read_u32(&mem, 0), 15);
    assert_eq!(cpu.read_gpr(12), 0x1122_3344_5566_7788);
    assert_eq!(read_u64(&mem, 8), 0x8877_6655_4433_2211);
}

#[test]
fn task47_rdtime_reads_monotonic_counter_and_tid() {
    let mut cpu = LoongArchCpu::new();
    cpu.csr_write(machina_guest_loongarch::loongarch::csr::CSR_TID, 7);

    let insns = [
        r2(OP_RDTIME_D, 2, 1),
        r2(OP_RDTIMEL_W, 4, 3),
        r2(OP_RDTIMEH_W, 6, 5),
    ];

    assert_eq!(run_la(&mut cpu, &insns), 0);
    assert_ne!(cpu.read_gpr(1), 0);
    assert_ne!(cpu.read_gpr(3), 0);
    assert_eq!(cpu.read_gpr(5), 0);
    assert_eq!(cpu.read_gpr(2), 7);
    assert_eq!(cpu.read_gpr(4), 7);
    assert_eq!(cpu.read_gpr(6), 7);
}

#[test]
fn task47_revh_variants_swap_halfword_order_for_fdt_parsing() {
    let mut cpu = LoongArchCpu::new();
    cpu.write_gpr(2, 0x1122_3344_5566_7788);
    cpu.write_gpr(4, 0x0000_0000_1122_3344);

    let insns = [r2(OP_REVH_D, 2, 3), r2(OP_REVH_2W, 4, 5)];

    assert_eq!(run_la(&mut cpu, &insns), 0);
    assert_eq!(cpu.read_gpr(3), 0x7788_5566_3344_1122);
    assert_eq!(cpu.read_gpr(5), 0x3344_1122);
}

#[test]
fn loongarch_barriers_preserve_registers_and_ibar_stops_tb() {
    let mut cpu = LoongArchCpu::new();
    let dbar_program = [
        r2_si12(OP_ADDI_D, 1, 0, 1),
        code15(OP_DBAR, 0),
        r2_si12(OP_ADDI_D, 2, 0, 1),
    ];

    assert_eq!(run_la(&mut cpu, &dbar_program), 0);
    assert_eq!(cpu.read_gpr(1), 2);

    let mut cpu = LoongArchCpu::new();
    let ibar_program = [
        r2_si12(OP_ADDI_D, 1, 0, 1),
        code15(OP_IBAR, 0),
        r2_si12(OP_ADDI_D, 2, 0, 1),
    ];

    assert_eq!(run_la(&mut cpu, &ibar_program), 0);
    assert_eq!(cpu.read_gpr(1), 1);
    assert_eq!(cpu.pc(), 8);
}

#[test]
fn task16_normal_ertn_invalidates_ll_sc_reservation() {
    let mut mem = [0u8; 64];
    mem[8..12].copy_from_slice(&0xAAAA_AAAAu32.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.set_ram_base(0);
    cpu.set_ram_end(mem.len() as u64);
    cpu.write_gpr(2, 0);

    assert_eq!(run_la(&mut cpu, &[r2_si14(OP_LL_W, 2, 2, 5)]), 0);
    assert_eq!(cpu.read_gpr(5), 0xFFFF_FFFF_AAAA_AAAA);
    cpu.write_gpr(5, 0x1111_1111);
    cpu.csr_write(CSR_CRMD, 0);
    cpu.csr_write(CSR_PRMD, 0);
    cpu.csr_write(CSR_ERA, 0);
    cpu.csr_write(CSR_TLBRERA, 0);

    let _ = run_la(&mut cpu, &[ERTN_INSN]);
    assert_eq!(run_la(&mut cpu, &[r2_si14(OP_SC_W, 2, 2, 5)]), 0);

    assert_eq!(cpu.read_gpr(5), 0);
    assert_eq!(read_u32(&mem, 8), 0xAAAA_AAAA);
}

#[test]
fn task16_tlbr_ertn_invalidates_ll_sc_reservation() {
    let mut mem = [0u8; 64];
    mem[8..12].copy_from_slice(&0xBBBB_BBBBu32.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.set_ram_base(0);
    cpu.set_ram_end(mem.len() as u64);
    cpu.write_gpr(2, 0);

    assert_eq!(run_la(&mut cpu, &[r2_si14(OP_LL_W, 2, 2, 5)]), 0);
    assert_eq!(cpu.read_gpr(5), 0xFFFF_FFFF_BBBB_BBBB);
    cpu.write_gpr(5, 0x2222_2222);
    cpu.csr_write(CSR_CRMD, CRMD_DA);
    cpu.csr_write(CSR_TLBRERA, 0x4000 | 1);
    cpu.csr_write(CSR_TLBRPRMD, 0);

    let _ = run_la(&mut cpu, &[ERTN_INSN]);
    assert_eq!(run_la(&mut cpu, &[r2_si14(OP_SC_W, 2, 2, 5)]), 0);

    assert_eq!(cpu.read_gpr(5), 0);
    assert_eq!(read_u32(&mem, 8), 0xBBBB_BBBB);
}

#[test]
fn task81_atomic_wd_overlap_rejects_rd_rj() {
    let mut mem = [0u8; 16];
    mem[0..4].copy_from_slice(&10u32.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.write_gpr(2, 0);
    cpu.write_gpr(3, 5);

    assert_eq!(
        run_la(&mut cpu, &[r3(OP_AMADD_W, 3, 2, 2)]),
        EXCP_UNDEF as usize
    );
    assert_eq!(read_u32(&mem, 0), 10);
}

#[test]
fn task81_atomic_wd_overlap_rejects_rd_rk_for_db_alias() {
    let mut mem = [0u8; 16];
    mem[8..16].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.write_gpr(2, 8);
    cpu.write_gpr(3, 0x8877_6655_4433_2211);

    assert_eq!(
        run_la(&mut cpu, &[r3(OP_AMSWAP_DB_D, 3, 2, 3)]),
        EXCP_UNDEF as usize
    );
    assert_eq!(read_u64(&mem, 8), 0x1122_3344_5566_7788);
}

#[test]
fn task81_atomic_rd_zero_overlap_is_legal_and_suppressed() {
    let mut mem = [0u8; 16];
    mem[0..4].copy_from_slice(&10u32.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.write_gpr(2, 0);
    cpu.write_gpr(3, 5);

    assert_eq!(run_la(&mut cpu, &[r3(OP_AMADD_W, 3, 2, 0)]), 0);
    assert_eq!(read_u32(&mem, 0), 15);
    assert_eq!(cpu.read_gpr(0), 0);
}

#[test]
fn task81_atomic_non_overlap_still_returns_old_value() {
    let mut mem = [0u8; 16];
    mem[8..16].copy_from_slice(&10u64.to_le_bytes());

    let mut cpu = LoongArchCpu::new();
    cpu.set_guest_base(mem.as_mut_ptr() as u64);
    cpu.write_gpr(2, 8);
    cpu.write_gpr(3, 5);

    assert_eq!(run_la(&mut cpu, &[r3(OP_AMADD_D, 3, 2, 4)]), 0);
    assert_eq!(cpu.read_gpr(4), 10);
    assert_eq!(read_u64(&mem, 8), 15);
}

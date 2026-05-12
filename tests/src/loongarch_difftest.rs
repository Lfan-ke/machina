use std::io::Write;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use machina_accel::code_buffer::CodeBuffer;
use machina_accel::ir::Context;
use machina_accel::translate::translate_and_execute;
use machina_accel::{HostCodeGen, X86_64CodeGen};
use machina_guest_loongarch::loongarch::cpu::{
    LoongArchCpu, GUEST_BASE_CPU_OFFSET, NUM_GPRS,
};
use machina_guest_loongarch::loongarch::ext::LoongArchCfg;
use machina_guest_loongarch::loongarch::trans::{
    LoongArchDisasContext, LoongArchTranslator,
};
use machina_guest_loongarch::translator_loop;

const SAVE_REG: usize = 12;
const MEMORY_BASE_REG: usize = 20;

const OP_ADD_D: u32 = 0b00000000000100001;
const OP_SUB_D: u32 = 0b00000000000100011;
const OP_SLT: u32 = 0b00000000000100100;
const OP_SLTU: u32 = 0b00000000000100101;
const OP_AND: u32 = 0b00000000000101001;
const OP_ORN: u32 = 0b00000000000101100;
const OP_DIV_D: u32 = 0b00000000001000100;
const OP_MOD_D: u32 = 0b00000000001000101;
const OP_DIV_WU: u32 = 0b00000000001000010;
const OP_MULW_D_W: u32 = 0b00000000000111110;
const OP_MULW_D_WU: u32 = 0b00000000000111111;
const OP_ROTR_D: u32 = 0b00000000000110111;
const OP_ALSL_D: u32 = 0b000000000010110;
const OP_MASKEQZ: u32 = 0b00000000000100110;
const OP_ADD_W: u32 = 0b00000000000100000;
const OP_ADDI_D: u32 = 0b0000001011;
const OP_ADDU16I_D: u32 = 0b000100;
const OP_CLZ_D: u32 = 0b0000000000000000001001;
const OP_BITREV_D: u32 = 0b0000000000000000010101;
const OP_BYTEPICK_W: u32 = 0b000000000000100;
const OP_BYTEPICK_D: u32 = 0b00000000000011;
const OP_EXT_W_B: u32 = 0b0000000000000000010111;
const OP_BSTR_W: u32 = 0b00000000011;
const OP_BSTRINS_D: u32 = 0b0000000010;
const OP_BSTRPICK_D: u32 = 0b0000000011;
const OP_LU12I_W: u32 = 0b0001010;
const OP_LU32I_D: u32 = 0b0001011;
const OP_LU52I_D: u32 = 0b0000001100;
const OP_LD_B: u32 = 0b0010100000;
const OP_LD_H: u32 = 0b0010100001;
const OP_LD_W: u32 = 0b0010100010;
const OP_LD_D: u32 = 0b0010100011;
const OP_ST_B: u32 = 0b0010100100;
const OP_ST_H: u32 = 0b0010100101;
const OP_ST_W: u32 = 0b0010100110;
const OP_ST_D: u32 = 0b0010100111;
const OP_LD_BU: u32 = 0b0010101000;
const OP_LD_HU: u32 = 0b0010101001;
const OP_LD_WU: u32 = 0b0010101010;
const OP_PRELD: u32 = 0b0010101011;
const OP_LDPTR_W: u32 = 0b00100100;
const OP_STPTR_W: u32 = 0b00100101;
const OP_LDPTR_D: u32 = 0b00100110;
const OP_STPTR_D: u32 = 0b00100111;
const OP_LDGT_B: u32 = 0b00111000011110000;
const OP_LDLE_B: u32 = 0b00111000011110100;
const OP_STLE_B: u32 = 0b00111000011111100;
const OP_AMSWAP_D: u32 = 0b00111000011000001;
const OP_AMADD_W: u32 = 0b00111000011000010;
const OP_AMMAX_WU: u32 = 0b00111000011001110;
const OP_AMMIN_DU: u32 = 0b00111000011010001;
const OP_DBAR: u32 = 0b00111000011100100;
const OP_IBAR: u32 = 0b00111000011100101;
const OP_BEQ: u32 = 0b010110;
const OP_BLT: u32 = 0b011000;
const OP_BGEU: u32 = 0b011011;
const OP_JIRL: u32 = 0b010011;
const OP_BEQZ: u32 = 0b010000;
const OP_BNEZ: u32 = 0b010001;
const OP_B: u32 = 0b010100;
const OP_BL: u32 = 0b010101;

#[derive(Clone)]
struct LoongArchDifftestCase {
    name: &'static str,
    init: Vec<(usize, u64)>,
    init_mem: Vec<(usize, u8)>,
    machina_insns: Vec<u32>,
    qemu_asm: &'static [&'static str],
    compare: &'static [usize],
}

fn r3(op: u32, rk: u32, rj: u32, rd: u32) -> u32 {
    (op << 15) | (rk << 10) | (rj << 5) | rd
}

fn r3_sa2(op: u32, sa2: u32, rk: u32, rj: u32, rd: u32) -> u32 {
    (op << 17) | (sa2 << 15) | (rk << 10) | (rj << 5) | rd
}

fn r3_sa3(op: u32, sa3: u32, rk: u32, rj: u32, rd: u32) -> u32 {
    (op << 18) | (sa3 << 15) | (rk << 10) | (rj << 5) | rd
}

fn r2_si16(op: u32, si16: i16, rj: u32, rd: u32) -> u32 {
    (op << 26) | ((si16 as u16 as u32) << 10) | (rj << 5) | rd
}

fn r1_offs21(op: u32, offs21: i32, rj: u32) -> u32 {
    let imm = offs21 as u32 & 0x001F_FFFF;
    (op << 26)
        | (((imm >> 16) & 0x1F) << 0)
        | ((imm & 0xFFFF) << 10)
        | (rj << 5)
}

fn offs26(op: u32, offs26: i32) -> u32 {
    let imm = offs26 as u32 & 0x03FF_FFFF;
    (op << 26) | (((imm >> 16) & 0x3FF) << 0) | ((imm & 0xFFFF) << 10)
}

fn r1_si20(op: u32, si20: i32, rd: u32) -> u32 {
    (op << 25) | ((si20 as u32 & 0x000F_FFFF) << 5) | rd
}

fn r2_si12(op: u32, si12: i16, rj: u32, rd: u32) -> u32 {
    (op << 22) | ((si12 as u16 as u32 & 0x0FFF) << 10) | (rj << 5) | rd
}

fn r2_si14(op: u32, si14: i16, rj: u32, rd: u32) -> u32 {
    (op << 24) | ((si14 as u16 as u32 & 0x3FFF) << 10) | (rj << 5) | rd
}

fn r2(op: u32, rj: u32, rd: u32) -> u32 {
    (op << 10) | (rj << 5) | rd
}

fn bstr_w(pick: bool, ms: u32, ls: u32, rj: u32, rd: u32) -> u32 {
    (OP_BSTR_W << 21)
        | ((ms & 0x1F) << 16)
        | (u32::from(pick) << 15)
        | ((ls & 0x1F) << 10)
        | (rj << 5)
        | rd
}

fn bstr_d(op: u32, ms: u32, ls: u32, rj: u32, rd: u32) -> u32 {
    (op << 22) | ((ms & 0x3F) << 16) | ((ls & 0x3F) << 10) | (rj << 5) | rd
}

fn code15(op: u32, code: u32) -> u32 {
    (op << 15) | (code & 0x7FFF)
}

fn build_qemu_asm(test: &LoongArchDifftestCase) -> String {
    let mut asm = String::from(
        ".text\n.global _start\n_start:\n    la.local $r12, save_area\n",
    );

    for &(reg, val) in &test.init {
        assert_ne!(reg, SAVE_REG, "r12 is reserved for the save-area pointer");
        assert!(reg < NUM_GPRS, "invalid GPR index {reg}");
        if reg != 0 {
            asm.push_str(&format!("    li.d $r{reg}, {}\n", val as i64));
        }
    }

    for &(off, val) in &test.init_mem {
        assert!(off < 256, "init memory offset {off} outside save_area");
        asm.push_str(&format!("    li.w $r30, {val}\n"));
        asm.push_str(&format!("    st.b $r30, $r12, {off}\n"));
    }
    if !test.init_mem.is_empty() {
        asm.push_str(&format!("    move $r{MEMORY_BASE_REG}, $r12\n"));
    }

    for line in test.qemu_asm {
        asm.push_str("    ");
        asm.push_str(line);
        asm.push('\n');
    }

    for reg in 0..NUM_GPRS {
        asm.push_str(&format!("    st.d $r{reg}, $r12, {}\n", reg * 8));
    }
    asm.push_str(
        "    li.w $a7, 64\n\
         \x20   li.w $a0, 1\n\
         \x20   move $a1, $r12\n\
         \x20   li.w $a2, 256\n\
         \x20   syscall 0\n\
         \x20   li.w $a7, 93\n\
         \x20   li.w $a0, 0\n\
         \x20   syscall 0\n\
         \x20   .bss\n\
         \x20   .align 3\n\
         save_area: .space 256\n",
    );
    asm
}

fn check_loongarch_difftest_toolchain() -> Result<(), String> {
    let qemu = Command::new("qemu-loongarch64")
        .arg("--version")
        .output()
        .map_err(|e| format!("qemu-loongarch64 is unavailable: {e}"))?;
    if !qemu.status.success() {
        return Err(format!(
            "qemu-loongarch64 --version failed:\n{}",
            String::from_utf8_lossy(&qemu.stderr)
        ));
    }

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir();
    let tag = format!(
        "machina_loongarch_toolchain_probe_{}_{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    let s_path = dir.join(format!("{tag}.S"));
    let obj_path = dir.join(format!("{tag}.o"));
    let cleanup = || {
        let _ = std::fs::remove_file(&s_path);
        let _ = std::fs::remove_file(&obj_path);
    };

    let mut file = std::fs::File::create(&s_path)
        .map_err(|e| format!("failed to create LoongArch probe source: {e}"))?;
    file.write_all(
        b".text\n.global _start\n_start:\n    add.d $r0, $r0, $r0\n",
    )
    .map_err(|e| format!("failed to write LoongArch probe source: {e}"))?;
    drop(file);

    let cc = Command::new("clang")
        .args([
            "--target=loongarch64-linux-gnu",
            "-c",
            "-o",
            obj_path.to_str().unwrap(),
            s_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("failed to run clang for LoongArch probe: {e}"))?;
    cleanup();
    if !cc.status.success() {
        return Err(format!(
            "clang does not support the LoongArch assembler target:\n{}",
            String::from_utf8_lossy(&cc.stderr)
        ));
    }

    Ok(())
}

fn run_qemu(test: &LoongArchDifftestCase) -> [u64; NUM_GPRS] {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    let dir = std::env::temp_dir();
    let tag = format!(
        "machina_loongarch_difftest_{}_{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    let s_path = dir.join(format!("{tag}.S"));
    let elf_path = dir.join(format!("{tag}.elf"));

    {
        let mut f = std::fs::File::create(&s_path).unwrap();
        f.write_all(build_qemu_asm(test).as_bytes()).unwrap();
    }

    let cc = Command::new("clang")
        .args([
            "--target=loongarch64-linux-gnu",
            "-fuse-ld=lld",
            "-nostdlib",
            "-static",
            "-o",
            elf_path.to_str().unwrap(),
            s_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run clang for LoongArch difftest");
    assert!(
        cc.status.success(),
        "clang failed for {}:\n{}",
        test.name,
        String::from_utf8_lossy(&cc.stderr)
    );

    let qemu = Command::new("qemu-loongarch64")
        .arg(elf_path.to_str().unwrap())
        .output()
        .expect("failed to run qemu-loongarch64");
    assert!(
        qemu.status.success(),
        "qemu-loongarch64 exited with {:?} for {}\nstderr:\n{}",
        qemu.status.code(),
        test.name,
        String::from_utf8_lossy(&qemu.stderr)
    );
    assert_eq!(
        qemu.stdout.len(),
        NUM_GPRS * 8,
        "expected {} bytes from qemu register dump for {}, got {}",
        NUM_GPRS * 8,
        test.name,
        qemu.stdout.len()
    );

    let mut regs = [0u64; NUM_GPRS];
    for (reg, slot) in regs.iter_mut().enumerate() {
        let off = reg * 8;
        *slot =
            u64::from_le_bytes(qemu.stdout[off..off + 8].try_into().unwrap());
    }

    let _ = std::fs::remove_file(&s_path);
    let _ = std::fs::remove_file(&elf_path);

    regs
}

fn run_machina(test: &LoongArchDifftestCase) -> [u64; NUM_GPRS] {
    let code: Vec<u8> = test
        .machina_insns
        .iter()
        .flat_map(|insn| insn.to_le_bytes())
        .collect();
    let code_len = code.len() as u64;

    let mut cpu = LoongArchCpu::new();
    let mut guest_mem = [0u8; 256];
    for &(off, val) in &test.init_mem {
        assert!(
            off < guest_mem.len(),
            "init memory offset {off} outside guest_mem"
        );
        guest_mem[off] = val;
    }
    cpu.set_guest_base(guest_mem.as_mut_ptr() as u64);
    if !test.init_mem.is_empty() {
        cpu.write_gpr(MEMORY_BASE_REG, 0);
    }
    for &(reg, val) in &test.init {
        cpu.write_gpr(reg, val);
    }

    for _ in 0..(test.machina_insns.len() + 8) {
        let pc = cpu.pc();
        if pc >= code_len {
            break;
        }
        assert_eq!(
            pc & 3,
            0,
            "Machina PC is not instruction-aligned for {}: {pc:#x}",
            test.name
        );
        assert!(
            pc + 4 <= code_len,
            "Machina PC outside code for {}: pc={pc:#x}, len={code_len:#x}",
            test.name
        );

        let mut backend = X86_64CodeGen::new();
        backend.set_guest_base_offset(GUEST_BASE_CPU_OFFSET);
        let mut buf = CodeBuffer::new(4096).unwrap();
        backend.emit_prologue(&mut buf);
        backend.emit_epilogue(&mut buf);

        let mut ir = Context::new();
        backend.init_context(&mut ir);

        let mut ctx = LoongArchDisasContext::new(
            pc,
            code.as_ptr(),
            LoongArchCfg::default(),
        );
        ctx.base.max_insns = ((code_len - pc) / 4) as u32;
        translator_loop::<LoongArchTranslator>(&mut ctx, &mut ir);

        let exit = unsafe {
            translate_and_execute(&mut ir, &backend, &mut buf, cpu.env_ptr())
        };
        assert!(
            exit <= 2,
            "Machina TB exited unexpectedly for {}: exit={exit}",
            test.name
        );
    }
    assert!(
        cpu.pc() >= code_len,
        "Machina did not reach end for {}: pc={:#x}, len={:#x}",
        test.name,
        cpu.pc(),
        code_len
    );

    let mut regs = [0u64; NUM_GPRS];
    for (reg, slot) in regs.iter_mut().enumerate() {
        *slot = cpu.read_gpr(reg);
    }
    regs
}

fn difftest_case(test: &LoongArchDifftestCase) {
    let qemu = run_qemu(test);
    let machina = run_machina(test);
    for &reg in test.compare {
        assert_eq!(
            machina[reg], qemu[reg],
            "LoongArch difftest mismatch for {} r{}: machina={:#x}, qemu={:#x}",
            test.name, reg, machina[reg], qemu[reg]
        );
    }
}

#[test]
fn loongarch_difftest_task6_task10_integer_matrix() {
    if let Err(reason) = check_loongarch_difftest_toolchain() {
        eprintln!("skipping LoongArch QEMU difftest: {reason}");
        return;
    }

    let cases = vec![
        LoongArchDifftestCase {
            name: "round10 arithmetic helpers",
            init: vec![
                (2, 0xFFFF_FFFF),
                (3, 2),
                (4, 0x1_0000),
                (5, 5),
                (6, 7),
                (7, 0x55AA),
                (8, 0),
            ],
            init_mem: vec![],
            machina_insns: vec![
                r3(OP_MULW_D_W, 3, 2, 1),
                r3(OP_MULW_D_WU, 3, 2, 9),
                r2_si16(OP_ADDU16I_D, -2, 4, 10),
                r3_sa2(OP_ALSL_D, 2, 6, 5, 11),
                r3(OP_MASKEQZ, 8, 7, 13),
            ],
            qemu_asm: &[
                "mulw.d.w $r1, $r2, $r3",
                "mulw.d.wu $r9, $r2, $r3",
                "addu16i.d $r10, $r4, -2",
                "alsl.d $r11, $r5, $r6, 3",
                "maskeqz $r13, $r7, $r8",
            ],
            compare: &[1, 9, 10, 11, 13],
        },
        LoongArchDifftestCase {
            name: "round11 division edges",
            init: vec![
                (2, 123),
                (3, 0),
                (4, i64::MIN as u64),
                (5, (-1i64) as u64),
                (6, 0x8000_0000),
            ],
            init_mem: vec![],
            machina_insns: vec![
                r3(OP_DIV_D, 3, 2, 1),
                r3(OP_MOD_D, 3, 2, 7),
                r3(OP_DIV_D, 5, 4, 8),
                r3(OP_MOD_D, 5, 4, 9),
                r3(OP_DIV_WU, 3, 6, 10),
            ],
            qemu_asm: &[
                "div.d $r1, $r2, $r3",
                "mod.d $r7, $r2, $r3",
                "div.d $r8, $r4, $r5",
                "mod.d $r9, $r4, $r5",
                "div.wu $r10, $r6, $r3",
            ],
            compare: &[1, 7, 8, 9, 10],
        },
        LoongArchDifftestCase {
            name: "round12 immediate loads",
            init: vec![],
            init_mem: vec![],
            machina_insns: vec![
                r1_si20(OP_LU12I_W, 0x12345, 1),
                r1_si20(OP_LU32I_D, 0x45678, 1),
                r2_si12(OP_LU52I_D, 0x123, 1, 1),
                r1_si20(OP_LU12I_W, 0x12345, 0),
            ],
            qemu_asm: &[
                "lu12i.w $r1, 0x12345",
                "lu32i.d $r1, 0x45678",
                "lu52i.d $r1, $r1, 0x123",
                "lu12i.w $r0, 0x12345",
            ],
            compare: &[0, 1],
        },
        LoongArchDifftestCase {
            name: "round13 bitfields",
            init: vec![
                (2, 0xFEDC_BA98_7654_3210),
                (4, 0x8000_1234),
                (5, 0xFFFF_0000_0000_0000),
                (6, 0x123),
                (9, 0x80),
                (18, 0x1122_3344_5566_7788),
                (19, 0x99AA_BBCC_DDEE_FF00),
            ],
            init_mem: vec![],
            machina_insns: vec![
                bstr_d(OP_BSTRPICK_D, 63, 63, 2, 1),
                bstr_w(true, 31, 0, 4, 7),
                bstr_d(OP_BSTRINS_D, 11, 4, 6, 5),
                r2(OP_EXT_W_B, 9, 8),
                r3_sa2(OP_BYTEPICK_W, 1, 19, 18, 20),
                r3_sa3(OP_BYTEPICK_D, 4, 19, 18, 21),
                r3_sa2(OP_BYTEPICK_W, 0, 19, 18, 22),
                r3_sa3(OP_BYTEPICK_D, 0, 19, 18, 23),
            ],
            qemu_asm: &[
                "bstrpick.d $r1, $r2, 63, 63",
                "bstrpick.w $r7, $r4, 31, 0",
                "bstrins.d $r5, $r6, 11, 4",
                "ext.w.b $r8, $r9",
                "bytepick.w $r20, $r18, $r19, 1",
                "bytepick.d $r21, $r18, $r19, 4",
                "bytepick.w $r22, $r18, $r19, 0",
                "bytepick.d $r23, $r18, $r19, 0",
            ],
            compare: &[1, 5, 7, 8, 20, 21, 22, 23],
        },
        LoongArchDifftestCase {
            name: "round14 broader alu matrix",
            init: vec![
                (2, 0x7FFF_FFFF),
                (3, 1),
                (4, 0b1100),
                (5, 0b1010),
                (8, (-1i64) as u64),
                (9, 1),
                (13, 1),
                (14, 65),
            ],
            init_mem: vec![],
            machina_insns: vec![
                r3(OP_ADD_W, 3, 2, 1),
                r3(OP_AND, 5, 4, 6),
                r3(OP_ORN, 5, 4, 7),
                r3(OP_SLT, 9, 8, 10),
                r3(OP_SLTU, 9, 8, 11),
                r3(OP_ROTR_D, 14, 13, 15),
                r2(OP_CLZ_D, 13, 16),
                r2(OP_BITREV_D, 13, 17),
                r3(OP_ADD_D, 3, 2, 0),
            ],
            qemu_asm: &[
                "add.w $r1, $r2, $r3",
                "and $r6, $r4, $r5",
                "orn $r7, $r4, $r5",
                "slt $r10, $r8, $r9",
                "sltu $r11, $r8, $r9",
                "rotr.d $r15, $r13, $r14",
                "clz.d $r16, $r13",
                "bitrev.d $r17, $r13",
                "add.d $r0, $r2, $r3",
            ],
            compare: &[0, 1, 6, 7, 10, 11, 15, 16, 17],
        },
        LoongArchDifftestCase {
            name: "round16 aligned load store matrix",
            init: vec![
                (10, 0xAA),
                (11, 0xBEEF),
                (13, 0x1122_3344),
                (14, 0x5566_7788_99AA_BBCC),
                (17, 0x0123_4567_89AB_CDEF),
                (18, 0x0000_0000_8000_0001),
            ],
            init_mem: vec![
                (0, 0x80),
                (2, 0x01),
                (3, 0x80),
                (4, 0x02),
                (5, 0x00),
                (6, 0x00),
                (7, 0x80),
                (8, 0x88),
                (9, 0x77),
                (10, 0x66),
                (11, 0x55),
                (12, 0x44),
                (13, 0x33),
                (14, 0x22),
                (15, 0x11),
            ],
            machina_insns: vec![
                r2_si12(OP_LD_B, 0, MEMORY_BASE_REG as u32, 1),
                r2_si12(OP_LD_H, 2, MEMORY_BASE_REG as u32, 2),
                r2_si12(OP_LD_W, 4, MEMORY_BASE_REG as u32, 3),
                r2_si12(OP_LD_D, 8, MEMORY_BASE_REG as u32, 4),
                r2_si12(OP_LD_BU, 0, MEMORY_BASE_REG as u32, 5),
                r2_si12(OP_LD_HU, 2, MEMORY_BASE_REG as u32, 6),
                r2_si12(OP_LD_WU, 4, MEMORY_BASE_REG as u32, 7),
                r2_si12(OP_ST_B, 24, MEMORY_BASE_REG as u32, 10),
                r2_si12(OP_LD_BU, 24, MEMORY_BASE_REG as u32, 8),
                r2_si12(OP_ST_H, 26, MEMORY_BASE_REG as u32, 11),
                r2_si12(OP_LD_HU, 26, MEMORY_BASE_REG as u32, 9),
                r2_si12(OP_ST_W, 28, MEMORY_BASE_REG as u32, 13),
                r2_si12(OP_LD_WU, 28, MEMORY_BASE_REG as u32, 15),
                r2_si12(OP_ST_D, 32, MEMORY_BASE_REG as u32, 14),
                r2_si12(OP_LD_D, 32, MEMORY_BASE_REG as u32, 16),
                r2_si12(OP_LD_D, 8, MEMORY_BASE_REG as u32, 0),
                r2_si14(OP_STPTR_D, 10, MEMORY_BASE_REG as u32, 17),
                r2_si14(OP_LDPTR_D, 10, MEMORY_BASE_REG as u32, 19),
                r2_si14(OP_STPTR_W, 12, MEMORY_BASE_REG as u32, 18),
                r2_si14(OP_LDPTR_W, 12, MEMORY_BASE_REG as u32, 22),
            ],
            qemu_asm: &[
                "ld.b $r1, $r20, 0",
                "ld.h $r2, $r20, 2",
                "ld.w $r3, $r20, 4",
                "ld.d $r4, $r20, 8",
                "ld.bu $r5, $r20, 0",
                "ld.hu $r6, $r20, 2",
                "ld.wu $r7, $r20, 4",
                "st.b $r10, $r20, 24",
                "ld.bu $r8, $r20, 24",
                "st.h $r11, $r20, 26",
                "ld.hu $r9, $r20, 26",
                "st.w $r13, $r20, 28",
                "ld.wu $r15, $r20, 28",
                "st.d $r14, $r20, 32",
                "ld.d $r16, $r20, 32",
                "ld.d $r0, $r20, 8",
                "stptr.d $r17, $r20, 40",
                "ldptr.d $r19, $r20, 40",
                "stptr.w $r18, $r20, 48",
                "ldptr.w $r22, $r20, 48",
            ],
            compare: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 19, 22],
        },
        LoongArchDifftestCase {
            name: "task81 preld and predicate memory success matrix",
            init: vec![(2, u64::MAX), (3, 0), (21, 8), (23, 0x44)],
            init_mem: vec![(0, 0x80), (8, 0x11)],
            machina_insns: vec![
                r2_si12(OP_PRELD, 0, MEMORY_BASE_REG as u32, 0),
                r3(OP_LDLE_B, 2, MEMORY_BASE_REG as u32, 1),
                r3(OP_LDGT_B, 3, 21, 4),
                r3(OP_STLE_B, 2, MEMORY_BASE_REG as u32, 23),
                r2_si12(OP_LD_BU, 0, MEMORY_BASE_REG as u32, 5),
            ],
            qemu_asm: &[
                "preld 0, $r20, 0",
                "ldle.b $r1, $r20, $r2",
                "addi.d $r21, $r12, 8",
                "ldgt.b $r4, $r21, $r3",
                "stle.b $r23, $r20, $r2",
                "ld.bu $r5, $r20, 0",
            ],
            compare: &[1, 4, 5],
        },
        LoongArchDifftestCase {
            name: "round19 atomic barrier matrix",
            init: vec![(10, 5), (11, 0x1122_3344_5566_7788), (13, 7), (14, 2)],
            init_mem: vec![
                (0, 0xFE),
                (1, 0xFF),
                (2, 0xFF),
                (3, 0xFF),
                (8, 10),
                (9, 0),
                (10, 0),
                (11, 0),
                (12, 0),
                (13, 0),
                (14, 0),
                (15, 0),
                (16, 0x01),
                (17, 0),
                (18, 0),
                (19, 0x80),
                (24, 0xFA),
                (25, 0xFF),
                (26, 0xFF),
                (27, 0xFF),
                (28, 0xFF),
                (29, 0xFF),
                (30, 0xFF),
                (31, 0xFF),
            ],
            machina_insns: vec![
                r3(OP_AMADD_W, 10, MEMORY_BASE_REG as u32, 1),
                r2_si12(OP_LD_WU, 0, MEMORY_BASE_REG as u32, 2),
                r2_si12(OP_ADDI_D, 8, MEMORY_BASE_REG as u32, 21),
                r3(OP_AMSWAP_D, 11, 21, 3),
                r2_si12(OP_LD_D, 0, 21, 4),
                r2_si12(OP_ADDI_D, 16, MEMORY_BASE_REG as u32, 22),
                r3(OP_AMMAX_WU, 14, 22, 5),
                r2_si12(OP_LD_WU, 0, 22, 6),
                r2_si12(OP_ADDI_D, 24, MEMORY_BASE_REG as u32, 23),
                r3(OP_AMMIN_DU, 13, 23, 7),
                r2_si12(OP_LD_D, 0, 23, 8),
                code15(OP_DBAR, 0),
                code15(OP_IBAR, 0),
                r2_si12(OP_ADDI_D, 7, 0, 9),
            ],
            qemu_asm: &[
                "amadd.w $r1, $r10, $r20",
                "ld.wu $r2, $r20, 0",
                "addi.d $r21, $r20, 8",
                "amswap.d $r3, $r11, $r21",
                "ld.d $r4, $r21, 0",
                "addi.d $r22, $r20, 16",
                "ammax.wu $r5, $r14, $r22",
                "ld.wu $r6, $r22, 0",
                "addi.d $r23, $r20, 24",
                "ammin.du $r7, $r13, $r23",
                "ld.d $r8, $r23, 0",
                "dbar 0",
                "ibar 0",
                "addi.d $r9, $r0, 7",
            ],
            compare: &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        LoongArchDifftestCase {
            name: "round21 memory branch combined matrix",
            init: vec![],
            init_mem: vec![(0, 5), (1, 0), (2, 0), (3, 0), (4, 0)],
            machina_insns: vec![
                r2_si12(OP_LD_WU, 0, MEMORY_BASE_REG as u32, 1),
                r2_si16(OP_BEQ, 4, 1, 0),
                r2_si12(OP_ADDI_D, 0x11, 0, 2),
                r2_si12(OP_ST_W, 24, MEMORY_BASE_REG as u32, 2),
                offs26(OP_B, 3),
                r2_si12(OP_ADDI_D, 0x99, 0, 2),
                r2_si12(OP_ST_W, 24, MEMORY_BASE_REG as u32, 2),
                r2_si12(OP_LD_WU, 4, MEMORY_BASE_REG as u32, 3),
                r2_si16(OP_BEQ, 4, 3, 0),
                r2_si12(OP_ADDI_D, 0x33, 0, 4),
                r2_si12(OP_ST_W, 28, MEMORY_BASE_REG as u32, 4),
                offs26(OP_B, 3),
                r2_si12(OP_ADDI_D, 0x22, 0, 4),
                r2_si12(OP_ST_W, 28, MEMORY_BASE_REG as u32, 4),
                r2_si12(OP_LD_WU, 24, MEMORY_BASE_REG as u32, 5),
                r2_si12(OP_LD_WU, 28, MEMORY_BASE_REG as u32, 6),
            ],
            qemu_asm: &[
                "ld.wu $r1, $r20, 0",
                "beq $r1, $r0, .Lround21_first_zero",
                "addi.d $r2, $r0, 0x11",
                "st.w $r2, $r20, 24",
                "b .Lround21_after_first",
                ".Lround21_first_zero:",
                "addi.d $r2, $r0, 0x99",
                "st.w $r2, $r20, 24",
                ".Lround21_after_first:",
                "ld.wu $r3, $r20, 4",
                "beq $r3, $r0, .Lround21_second_zero",
                "addi.d $r4, $r0, 0x33",
                "st.w $r4, $r20, 28",
                "b .Lround21_after_second",
                ".Lround21_second_zero:",
                "addi.d $r4, $r0, 0x22",
                "st.w $r4, $r20, 28",
                ".Lround21_after_second:",
                "ld.wu $r5, $r20, 24",
                "ld.wu $r6, $r20, 28",
            ],
            compare: &[1, 2, 3, 4, 5, 6],
        },
        LoongArchDifftestCase {
            name: "round21 branch selected atomic matrix",
            init: vec![(10, 7)],
            init_mem: vec![(0, 1), (8, 10), (9, 0), (10, 0), (11, 0)],
            machina_insns: vec![
                r2_si12(OP_LD_WU, 0, MEMORY_BASE_REG as u32, 1),
                r1_offs21(OP_BEQZ, 5, 1),
                r2_si12(OP_ADDI_D, 8, MEMORY_BASE_REG as u32, 21),
                r3(OP_AMADD_W, 10, 21, 5),
                r2_si12(OP_LD_WU, 0, 21, 6),
                offs26(OP_B, 3),
                r2_si12(OP_ADDI_D, 0, 0, 5),
                r2_si12(OP_LD_WU, 8, MEMORY_BASE_REG as u32, 6),
                r2_si12(OP_ST_W, 16, MEMORY_BASE_REG as u32, 6),
                r2_si12(OP_LD_WU, 16, MEMORY_BASE_REG as u32, 7),
            ],
            qemu_asm: &[
                "ld.wu $r1, $r20, 0",
                "beqz $r1, .Lround21_skip_am",
                "addi.d $r21, $r20, 8",
                "amadd.w $r5, $r10, $r21",
                "ld.wu $r6, $r21, 0",
                "b .Lround21_after_am",
                ".Lround21_skip_am:",
                "addi.d $r5, $r0, 0",
                "ld.wu $r6, $r20, 8",
                ".Lround21_after_am:",
                "st.w $r6, $r20, 16",
                "ld.wu $r7, $r20, 16",
            ],
            compare: &[1, 5, 6, 7],
        },
        LoongArchDifftestCase {
            name: "round17 branch jump matrix",
            init: vec![
                (2, 7),
                (3, 7),
                (4, 8),
                (6, (-2i64) as u64),
                (7, 1),
                (8, 1),
                (9, u64::MAX),
                (10, 0),
                (11, 9),
                (19, 104),
                (20, 120),
                (30, 96),
                (31, 80),
            ],
            init_mem: vec![],
            machina_insns: vec![
                r2_si16(OP_BEQ, 2, 2, 3),
                r2_si12(OP_ADDI_D, 11, 0, 21),
                r2_si12(OP_ADDI_D, 21, 0, 21),
                r2_si16(OP_BEQ, 3, 2, 4),
                r2_si12(OP_ADDI_D, 22, 0, 22),
                offs26(OP_B, 2),
                r2_si12(OP_ADDI_D, 99, 0, 22),
                r2_si16(OP_BLT, 2, 6, 7),
                r2_si12(OP_ADDI_D, 11, 0, 23),
                r2_si12(OP_ADDI_D, 23, 0, 23),
                r2_si16(OP_BGEU, 3, 8, 9),
                r2_si12(OP_ADDI_D, 24, 0, 24),
                offs26(OP_B, 2),
                r2_si12(OP_ADDI_D, 99, 0, 24),
                r1_offs21(OP_BEQZ, 2, 10),
                r2_si12(OP_ADDI_D, 11, 0, 25),
                r2_si12(OP_ADDI_D, 25, 0, 25),
                r1_offs21(OP_BNEZ, 2, 11),
                r2_si12(OP_ADDI_D, 11, 0, 26),
                r2_si12(OP_ADDI_D, 26, 0, 26),
                offs26(OP_BL, 2),
                r2_si12(OP_ADDI_D, 11, 0, 27),
                r3(OP_SUB_D, 31, 1, 1),
                r2_si12(OP_ADDI_D, 27, 0, 27),
                r2_si16(OP_JIRL, 0, 19, 18),
                r2_si12(OP_ADDI_D, 11, 0, 17),
                r3(OP_SUB_D, 30, 18, 18),
                r2_si12(OP_ADDI_D, 17, 0, 17),
                r2_si16(OP_JIRL, 0, 20, 0),
                r2_si12(OP_ADDI_D, 11, 0, 16),
                r2_si12(OP_ADDI_D, 16, 0, 16),
            ],
            qemu_asm: &[
                "beq $r2, $r3, .Lbeq_taken",
                "addi.d $r21, $r0, 11",
                ".Lbeq_taken:",
                "addi.d $r21, $r0, 21",
                "beq $r2, $r4, .Lbeq_not_expected",
                "addi.d $r22, $r0, 22",
                "b .Lafter_beq_not",
                ".Lbeq_not_expected:",
                "addi.d $r22, $r0, 99",
                ".Lafter_beq_not:",
                "blt $r6, $r7, .Lblt_taken",
                "addi.d $r23, $r0, 11",
                ".Lblt_taken:",
                "addi.d $r23, $r0, 23",
                "bgeu $r8, $r9, .Lbgeu_not_expected",
                "addi.d $r24, $r0, 24",
                "b .Lafter_bgeu_not",
                ".Lbgeu_not_expected:",
                "addi.d $r24, $r0, 99",
                ".Lafter_bgeu_not:",
                "beqz $r10, .Lbeqz_taken",
                "addi.d $r25, $r0, 11",
                ".Lbeqz_taken:",
                "addi.d $r25, $r0, 25",
                "bnez $r11, .Lbnez_taken",
                "addi.d $r26, $r0, 11",
                ".Lbnez_taken:",
                "addi.d $r26, $r0, 26",
                "la.local $r31, .Lbl_base",
                ".Lbl_base:",
                "bl .Lbl_target",
                "addi.d $r27, $r0, 11",
                ".Lbl_target:",
                "sub.d $r1, $r1, $r31",
                "addi.d $r27, $r0, 27",
                "la.local $r19, .Ljirl_target",
                "la.local $r30, .Ljirl_base",
                ".Ljirl_base:",
                "jirl $r18, $r19, 0",
                "addi.d $r17, $r0, 11",
                ".Ljirl_target:",
                "sub.d $r18, $r18, $r30",
                "addi.d $r17, $r0, 17",
                "la.local $r20, .Ljirl_zero_target",
                "jirl $r0, $r20, 0",
                "addi.d $r16, $r0, 11",
                ".Ljirl_zero_target:",
                "addi.d $r16, $r0, 16",
            ],
            compare: &[0, 1, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27],
        },
    ];

    for case in cases {
        difftest_case(&case);
    }
}

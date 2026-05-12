pub mod gen_common;
pub mod gen_float;
pub mod helpers;

use machina_accel::ir::tb::{EXCP_LOONGARCH_DONE, EXCP_UNDEF, TB_EXIT_IDX0};
use machina_accel::ir::{Context, TempIdx, Type};

use super::cpu::{gpr_offset, NUM_GPRS, PC_OFFSET};
use super::exception::{ECODE_BRK, ECODE_HVC, ECODE_SYS};
use super::ext::LoongArchCfg;
use super::insn_decode;
use crate::{DisasContextBase, DisasJumpType, TranslatorOps};

pub struct LoongArchDisasContext {
    pub base: DisasContextBase,
    pub cfg: LoongArchCfg,
    pub env: TempIdx,
    pub gpr: [TempIdx; NUM_GPRS],
    pub pc: TempIdx,
    pub llbctl: TempIdx,
    pub ll_res_addr: TempIdx,
    pub ll_res_val: TempIdx,
    pub opcode: u32,
    pub guest_base: *const u8,
}

impl LoongArchDisasContext {
    pub const GLOBAL_COUNT: u32 = 1 + NUM_GPRS as u32 + 1 + 3;

    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(pc: u64, guest_base: *const u8, cfg: LoongArchCfg) -> Self {
        Self {
            base: DisasContextBase {
                pc_first: pc,
                pc_next: pc,
                is_jmp: DisasJumpType::Next,
                num_insns: 0,
                max_insns: 512,
            },
            cfg,
            env: TempIdx(0),
            gpr: [TempIdx(0); NUM_GPRS],
            pc: TempIdx(0),
            llbctl: TempIdx(0),
            ll_res_addr: TempIdx(0),
            ll_res_val: TempIdx(0),
            opcode: 0,
            guest_base,
        }
    }

    pub fn bind_existing_globals(&mut self, ir: &Context) {
        assert_eq!(
            ir.nb_globals(),
            Self::GLOBAL_COUNT,
            "LoongArch translator global layout changed"
        );
        self.env = TempIdx(0);
        for i in 0..NUM_GPRS {
            self.gpr[i] = TempIdx((1 + i) as u32);
        }
        self.pc = TempIdx(1 + NUM_GPRS as u32);
        self.llbctl = TempIdx(2 + NUM_GPRS as u32);
        self.ll_res_addr = TempIdx(3 + NUM_GPRS as u32);
        self.ll_res_val = TempIdx(4 + NUM_GPRS as u32);
    }
}

fn gen_nanbox_s_value(ir: &mut Context, value: TempIdx) -> TempIdx {
    let low_mask = ir.new_const(Type::I64, 0xffff_ffff);
    let upper = ir.new_const(Type::I64, 0xffff_ffff_0000_0000);
    let low = ir.new_temp(Type::I64);
    ir.gen_and(Type::I64, low, value, low_mask);
    let result = ir.new_temp(Type::I64);
    ir.gen_or(Type::I64, result, low, upper);
    result
}

fn gen_fp_addr(
    ctx: &LoongArchDisasContext,
    ir: &mut Context,
    rj: u8,
    rk: u8,
) -> TempIdx {
    let base = gen_common::gpr_get(&ctx.gpr, ir, rj);
    let index = gen_common::gpr_get(&ctx.gpr, ir, rk);
    let addr = ir.new_temp(Type::I64);
    ir.gen_add(Type::I64, addr, base, index);
    addr
}

fn gen_index_addr(
    ctx: &LoongArchDisasContext,
    ir: &mut Context,
    rj: u8,
    rk: u8,
) -> TempIdx {
    let base = gen_common::gpr_get(&ctx.gpr, ir, rj);
    let index = gen_common::gpr_get(&ctx.gpr, ir, rk);
    let addr = ir.new_temp(Type::I64);
    ir.gen_add(Type::I64, addr, base, index);
    addr
}

fn gen_gload_addr(
    ctx: &LoongArchDisasContext,
    ir: &mut Context,
    rd: u8,
    addr: TempIdx,
    memop: machina_accel::ir::MemOp,
) {
    let value = ir.new_temp(Type::I64);
    ir.gen_qemu_ld(Type::I64, value, addr, u32::from(memop.bits()));
    gen_common::gpr_set(&ctx.gpr, ir, rd, value);
}

fn gen_gstore_addr(
    ctx: &LoongArchDisasContext,
    ir: &mut Context,
    rd: u8,
    addr: TempIdx,
    memop: machina_accel::ir::MemOp,
) {
    let value = gen_common::gpr_get(&ctx.gpr, ir, rd);
    ir.gen_qemu_st(Type::I64, value, addr, u32::from(memop.bits()));
}

fn gen_predicate_addr(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
    rj: u8,
    rk: u8,
    helper: unsafe extern "sysv64" fn(*mut u8, u64, u64, u64) -> u64,
) -> TempIdx {
    gen_fp_predicate_assert(ctx, ir, rj, rk, helper);
    gen_common::gpr_get(&ctx.gpr, ir, rj)
}

fn am_r3_has_forbidden_overlap(a: &insn_decode::ArgsR3) -> bool {
    a.rd != 0 && (a.rd == a.rj || a.rd == a.rk)
}

fn gen_rdtime(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
    rd: u8,
    rj: u8,
    word: bool,
    high: bool,
) {
    let time = ir.new_temp(Type::I64);
    ir.gen_call(
        time,
        helpers::loongarch_helper_rdtime_d as *const () as u64,
        &[ctx.env],
    );
    let value = if word {
        let shifted = if high {
            let shift = ir.new_const(Type::I64, 32);
            let tmp = ir.new_temp(Type::I64);
            ir.gen_shr(Type::I64, tmp, time, shift);
            tmp
        } else {
            time
        };
        let ext = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(ext, shifted);
        ext
    } else {
        time
    };
    gen_common::gpr_set(&ctx.gpr, ir, rd, value);

    let tid = ir.new_temp(Type::I64);
    ir.gen_call(
        tid,
        helpers::loongarch_helper_tid as *const () as u64,
        &[ctx.env],
    );
    gen_common::gpr_set(&ctx.gpr, ir, rj, tid);
}

fn gen_fp_predicate_assert(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
    rj: u8,
    rk: u8,
    helper: unsafe extern "sysv64" fn(*mut u8, u64, u64, u64) -> u64,
) {
    use machina_accel::ir::Cond;

    let src1 = gen_common::gpr_get(&ctx.gpr, ir, rj);
    let src2 = gen_common::gpr_get(&ctx.gpr, ir, rk);
    let pc = ir.new_const(Type::I64, ctx.base.pc_next - 4);
    let trap = ir.new_temp(Type::I64);
    ir.gen_call(trap, helper as *const () as u64, &[ctx.env, src1, src2, pc]);

    let zero = ir.new_const(Type::I64, 0);
    let label_ok = ir.new_label();
    ir.gen_brcond(Type::I64, trap, zero, Cond::Eq, label_ok);
    ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
    ir.gen_set_label(label_ok);
}

fn gen_guest_sensitive_check(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
) {
    use machina_accel::ir::Cond;

    let chk = ir.new_temp(Type::I64);
    ir.gen_call(
        chk,
        helpers::loongarch_helper_check_guest_sensitive as *const () as u64,
        &[ctx.env],
    );
    let zero = ir.new_const(Type::I64, 0);
    let label_ok = ir.new_label();
    ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
    ir.gen_mov(Type::I64, ctx.pc, chk);
    ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
    ir.gen_set_label(label_ok);
}

fn gen_guest_csr_check(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
    csr_num: u32,
) {
    use machina_accel::ir::Cond;

    let csr_num = ir.new_const(Type::I64, u64::from(csr_num));
    let chk = ir.new_temp(Type::I64);
    ir.gen_call(
        chk,
        helpers::loongarch_helper_check_guest_csr as *const () as u64,
        &[ctx.env, csr_num],
    );
    let zero = ir.new_const(Type::I64, 0);
    let label_ok = ir.new_label();
    ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
    ir.gen_mov(Type::I64, ctx.pc, chk);
    ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
    ir.gen_set_label(label_ok);
}

fn gen_fload_addr(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
    fd: u8,
    addr: TempIdx,
    memop: machina_accel::ir::MemOp,
    nanbox: bool,
) {
    let value = ir.new_temp(Type::I64);
    ir.gen_qemu_ld(Type::I64, value, addr, u32::from(memop.bits()));
    let value = if nanbox {
        gen_nanbox_s_value(ir, value)
    } else {
        value
    };
    gen_float::fpr_set(ctx, ir, fd, value);
}

fn gen_fstore_addr(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
    fd: u8,
    addr: TempIdx,
    memop: machina_accel::ir::MemOp,
) {
    let value = gen_float::fpr_get(ctx, ir, fd);
    ir.gen_qemu_st(Type::I64, value, addr, u32::from(memop.bits()));
}

pub struct LoongArchTranslator;

impl TranslatorOps for LoongArchTranslator {
    type DisasContext = LoongArchDisasContext;

    fn init_disas_context(ctx: &mut Self::DisasContext, ir: &mut Context) {
        ctx.env = ir.new_fixed(Type::I64, 5, "env");
        for i in 0..NUM_GPRS {
            ctx.gpr[i] = ir.new_global(
                Type::I64,
                ctx.env,
                i64::try_from(gpr_offset(i)).unwrap(),
                "gpr",
            );
        }
        ctx.pc = ir.new_global(
            Type::I64,
            ctx.env,
            i64::try_from(PC_OFFSET).unwrap(),
            "pc",
        );
        ctx.llbctl = ir.new_global(
            Type::I64,
            ctx.env,
            i64::try_from(super::cpu::LLBCTL_OFFSET).unwrap(),
            "llbctl",
        );
        ctx.ll_res_addr = ir.new_global(
            Type::I64,
            ctx.env,
            i64::try_from(super::cpu::LL_RES_ADDR_OFFSET).unwrap(),
            "ll_res_addr",
        );
        ctx.ll_res_val = ir.new_global(
            Type::I64,
            ctx.env,
            i64::try_from(super::cpu::LL_RES_VAL_OFFSET).unwrap(),
            "ll_res_val",
        );
        debug_assert_eq!(ir.nb_globals(), LoongArchDisasContext::GLOBAL_COUNT);
    }

    fn tb_start(_ctx: &mut Self::DisasContext, _ir: &mut Context) {}

    fn insn_start(ctx: &mut Self::DisasContext, ir: &mut Context) {
        ir.gen_insn_start(ctx.base.pc_next);
        ctx.base.num_insns += 1;
    }

    fn translate_insn(ctx: &mut Self::DisasContext, ir: &mut Context) {
        let pc = ctx.base.pc_next;
        // SAFETY: guest_base + pc must be a valid, readable 4-byte
        // host address. The caller (system crate or test harness)
        // ensures this by mapping guest memory before translation.
        let insn = unsafe {
            let ptr = ctx.guest_base.add(usize::try_from(pc).unwrap());
            ptr.cast::<u32>().read_unaligned()
        };
        ctx.opcode = insn;
        ctx.base.pc_next = pc + 4;

        if !decode_insn(ctx, ir, insn) {
            ctx.base.pc_next = pc;
            let c = ir.new_const(Type::I64, pc);
            ir.gen_mov(Type::I64, ctx.pc, c);
            ir.gen_exit_tb(EXCP_UNDEF);
            ctx.base.is_jmp = DisasJumpType::NoReturn;
        }
    }

    fn tb_stop(ctx: &mut Self::DisasContext, ir: &mut Context) {
        match ctx.base.is_jmp {
            DisasJumpType::Next | DisasJumpType::TooMany => {
                let c = ir.new_const(Type::I64, ctx.base.pc_next);
                ir.gen_mov(Type::I64, ctx.pc, c);
                ir.gen_goto_tb(0);
                ir.gen_exit_tb(TB_EXIT_IDX0);
            }
            DisasJumpType::NoReturn => {}
        }
    }

    fn base(ctx: &Self::DisasContext) -> &DisasContextBase {
        &ctx.base
    }

    fn base_mut(ctx: &mut Self::DisasContext) -> &mut DisasContextBase {
        &mut ctx.base
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
impl insn_decode::Decode<Context> for LoongArchDisasContext {
    fn trans_add_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_sub_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_sub(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_and(&mut self, ir: &mut Context, a: &insn_decode::ArgsR3) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_or(&mut self, ir: &mut Context, a: &insn_decode::ArgsR3) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_xor(&mut self, ir: &mut Context, a: &insn_decode::ArgsR3) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_xor(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_nor(&mut self, ir: &mut Context, a: &insn_decode::ArgsR3) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let t = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, t, s1, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_not(Type::I64, d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_slt(&mut self, ir: &mut Context, a: &insn_decode::ArgsR3) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_setcond(Type::I64, d, s1, s2, Cond::Lt);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_sltu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_setcond(Type::I64, d, s1, s2, Cond::Ltu);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_sll_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_srl_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_shr(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_sra_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_sar(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_addi_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let imm = ir.new_const(Type::I64, a.si12 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, d, src, imm);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_andi(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let imm = ir.new_const(Type::I64, a.ui12 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, d, src, imm);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ori(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let imm = ir.new_const(Type::I64, a.ui12 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, d, src, imm);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_xori(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let imm = ir.new_const(Type::I64, a.ui12 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_xor(Type::I64, d, src, imm);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_addu16i_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si16,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let imm = ir.new_const(Type::I64, (a.si16 << 16) as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, d, src, imm);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_lu12i_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR1Si20,
    ) -> bool {
        use gen_common::gpr_set;
        let val = (a.si20 << 12) as u64;
        let d = ir.new_const(Type::I64, val);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_lu32i_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR1Si20,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rd as u8);
        let mask = ir.new_const(Type::I64, 0xFFFF_FFFF);
        let low = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, low, src, mask);
        let hi = ir.new_const(Type::I64, (a.si20 << 32) as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, d, low, hi);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_pcalau12i(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR1Si20,
    ) -> bool {
        use gen_common::gpr_set;
        let pc_val = self.base.pc_next - 4;
        let result = pc_val.wrapping_add((a.si20 << 12) as u64) & !0xfff;
        let d = ir.new_const(Type::I64, result);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_pcaddu12i(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR1Si20,
    ) -> bool {
        use gen_common::gpr_set;
        let pc_val = self.base.pc_next - 4;
        let result = pc_val.wrapping_add((a.si20 << 12) as u64);
        let d = ir.new_const(Type::I64, result);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_pcaddu18i(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR1Si20,
    ) -> bool {
        use gen_common::gpr_set;
        let pc_val = self.base.pc_next - 4;
        let result = pc_val.wrapping_add((a.si20 << 18) as u64);
        let d = ir.new_const(Type::I64, result);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mul_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_mul(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mulw_d_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let s1_ext = ir.new_temp(Type::I64);
        let s2_ext = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(s1_ext, s1);
        ir.gen_ext_i32_i64(s2_ext, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_mul(Type::I64, d, s1_ext, s2_ext);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mulw_d_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let s1_ext = ir.new_temp(Type::I64);
        let s2_ext = ir.new_temp(Type::I64);
        ir.gen_ext_u32_i64(s1_ext, s1);
        ir.gen_ext_u32_i64(s2_ext, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_mul(Type::I64, d, s1_ext, s2_ext);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_div_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_div_d as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_div_du(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_div_du as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mod_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_mod_d as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mod_du(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_mod_du as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_div_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_div_w as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_div_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_div_wu as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mod_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_mod_w as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mod_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_mod_wu as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_add_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let t = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, t, s1, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_sub_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let t = ir.new_temp(Type::I64);
        ir.gen_sub(Type::I64, t, s1, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_addi_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let imm = ir.new_const(Type::I64, a.si12 as u64);
        let t = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, t, src, imm);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_slti(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let imm = ir.new_const(Type::I64, a.si12 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_setcond(Type::I64, d, src, imm, Cond::Lt);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_sltui(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let imm = ir.new_const(Type::I64, a.si12 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_setcond(Type::I64, d, src, imm, Cond::Ltu);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_orn(&mut self, ir: &mut Context, a: &insn_decode::ArgsR3) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let not_s2 = ir.new_temp(Type::I64);
        ir.gen_not(Type::I64, not_s2, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, d, s1, not_s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_andn(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let not_s2 = ir.new_temp(Type::I64);
        ir.gen_not(Type::I64, not_s2, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, d, s1, not_s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_maskeqz(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let test = gpr_get(&self.gpr, ir, a.rk as u8);
        let zero = ir.new_const(Type::I64, 0);
        let d = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, d, test, zero, zero, src, Cond::Eq);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_masknez(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let test = gpr_get(&self.gpr, ir, a.rk as u8);
        let zero = ir.new_const(Type::I64, 0);
        let d = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, d, test, zero, zero, src, Cond::Ne);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_lu52i_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let mask = ir.new_const(Type::I64, 0x000F_FFFF_FFFF_FFFF);
        let low = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, low, src, mask);
        let hi = ir.new_const(Type::I64, (a.si12 << 52) as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, d, low, hi);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_pcaddi(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR1Si20,
    ) -> bool {
        use gen_common::gpr_set;
        let pc_val = self.base.pc_next - 4;
        let result = pc_val.wrapping_add((a.si20 << 2) as u64);
        let d = ir.new_const(Type::I64, result);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ext_w_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, src);
        let mask_byte = ir.new_const(Type::I64, 0xFF);
        let byte_val = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, byte_val, src, mask_byte);
        let sext = ir.new_temp(Type::I64);
        let shift = ir.new_const(Type::I64, 56);
        ir.gen_shl(Type::I64, sext, byte_val, shift);
        let result = ir.new_temp(Type::I64);
        ir.gen_sar(Type::I64, result, sext, shift);
        gpr_set(&self.gpr, ir, a.rd as u8, result);
        true
    }

    fn trans_ext_w_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let mask_hw = ir.new_const(Type::I64, 0xFFFF);
        let hw_val = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, hw_val, src, mask_hw);
        let shift = ir.new_const(Type::I64, 48);
        let sext = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I64, sext, hw_val, shift);
        let result = ir.new_temp(Type::I64);
        ir.gen_sar(Type::I64, result, sext, shift);
        gpr_set(&self.gpr, ir, a.rd as u8, result);
        true
    }

    fn trans_alsl_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3Sa2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let shift = ir.new_const(Type::I64, (a.sa2 + 1) as u64);
        let shifted = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I64, shifted, s1, shift);
        let sum = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, sum, shifted, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, sum);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_alsl_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3Sa2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let shift = ir.new_const(Type::I64, (a.sa2 + 1) as u64);
        let shifted = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I64, shifted, s1, shift);
        let sum = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, sum, shifted, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_u32_i64(d, sum);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_alsl_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3Sa2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let shift = ir.new_const(Type::I64, (a.sa2 + 1) as u64);
        let shifted = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I64, shifted, s1, shift);
        let d = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, d, shifted, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mul_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let t = ir.new_temp(Type::I64);
        ir.gen_mul(Type::I64, t, s1, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_sll_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let t = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I32, t, s1, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_srl_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let t = ir.new_temp(Type::I64);
        ir.gen_shr(Type::I32, t, s1, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_sra_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let t = ir.new_temp(Type::I64);
        ir.gen_sar(Type::I32, t, s1, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_rotr_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let t = ir.new_temp(Type::I64);
        ir.gen_rotr(Type::I32, t, s1, s2);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_rotr_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_rotr(Type::I64, d, s1, s2);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bytepick_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3Sa2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let rj = gpr_get(&self.gpr, ir, a.rj as u8);
        let rk = gpr_get(&self.gpr, ir, a.rk as u8);
        let picked = ir.new_temp(Type::I64);
        let offset = 32 - (a.sa2 as u32) * 8;
        if offset == 32 {
            ir.gen_mov(Type::I64, picked, rk);
        } else {
            ir.gen_extract2(Type::I32, picked, rj, rk, offset);
        }
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, picked);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bytepick_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3Sa3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let rj = gpr_get(&self.gpr, ir, a.rj as u8);
        let rk = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        let offset = 64 - (a.sa3 as u32) * 8;
        if offset == 64 {
            ir.gen_mov(Type::I64, d, rk);
        } else {
            ir.gen_extract2(Type::I64, d, rj, rk, offset);
        }
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bstrpick_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Msbw,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let msbw = a.msbw as u64;
        let lsbw = a.lsbw as u64;
        if msbw < lsbw {
            return false;
        }
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let shift = ir.new_const(Type::I64, lsbw);
        let shifted = ir.new_temp(Type::I64);
        ir.gen_shr(Type::I64, shifted, src, shift);
        let width = msbw - lsbw + 1;
        let mask_val = (1u64 << width) - 1;
        let mask = ir.new_const(Type::I64, mask_val);
        let picked = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, picked, shifted, mask);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, picked);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bstrpick_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Msbd,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let msbd = a.msbd as u64;
        let lsbd = a.lsbd as u64;
        if msbd < lsbd {
            return false;
        }
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let shift = ir.new_const(Type::I64, lsbd);
        let shifted = ir.new_temp(Type::I64);
        ir.gen_shr(Type::I64, shifted, src, shift);
        let width = msbd - lsbd + 1;
        let mask_val = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let mask = ir.new_const(Type::I64, mask_val);
        let d = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, d, shifted, mask);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bstrins_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Msbw,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let msbw = a.msbw as u64;
        let lsbw = a.lsbw as u64;
        if msbw < lsbw {
            return false;
        }
        let width = msbw - lsbw + 1;
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let dst_old = gpr_get(&self.gpr, ir, a.rd as u8);
        let field_mask = (1u64 << width) - 1;
        let src_mask = ir.new_const(Type::I64, field_mask);
        let src_masked = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, src_masked, src, src_mask);
        let shift = ir.new_const(Type::I64, lsbw);
        let src_shifted = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I64, src_shifted, src_masked, shift);
        let clear_mask = ir.new_const(Type::I64, !(field_mask << lsbw));
        let dst_cleared = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, dst_cleared, dst_old, clear_mask);
        let merged = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, merged, dst_cleared, src_shifted);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, merged);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bstrins_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Msbd,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let msbd = a.msbd as u64;
        let lsbd = a.lsbd as u64;
        if msbd < lsbd {
            return false;
        }
        let width = msbd - lsbd + 1;
        let src = gpr_get(&self.gpr, ir, a.rj as u8);
        let dst_old = gpr_get(&self.gpr, ir, a.rd as u8);
        let field_mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let src_mask = ir.new_const(Type::I64, field_mask);
        let src_masked = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, src_masked, src, src_mask);
        let shift = ir.new_const(Type::I64, lsbd);
        let src_shifted = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I64, src_shifted, src_masked, shift);
        let clear_mask = ir.new_const(Type::I64, !(field_mask << lsbd));
        let dst_cleared = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, dst_cleared, dst_old, clear_mask);
        let d = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, d, dst_cleared, src_shifted);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    // --- Shift/rotate by immediate ---

    fn trans_slli_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui5,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let sh = ir.new_const(Type::I64, a.ui5 as u64);
        let t = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I32, t, s, sh);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_slli_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui6,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let sh = ir.new_const(Type::I64, a.ui6 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_shl(Type::I64, d, s, sh);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_srli_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui5,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let sh = ir.new_const(Type::I64, a.ui5 as u64);
        let t = ir.new_temp(Type::I64);
        ir.gen_shr(Type::I32, t, s, sh);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_srli_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui6,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let sh = ir.new_const(Type::I64, a.ui6 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_shr(Type::I64, d, s, sh);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_srai_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui5,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let sh = ir.new_const(Type::I64, a.ui5 as u64);
        let t = ir.new_temp(Type::I64);
        ir.gen_sar(Type::I32, t, s, sh);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_srai_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui6,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let sh = ir.new_const(Type::I64, a.ui6 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_sar(Type::I64, d, s, sh);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_rotri_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui5,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let sh = ir.new_const(Type::I64, a.ui5 as u64);
        let t = ir.new_temp(Type::I64);
        ir.gen_rotr(Type::I32, t, s, sh);
        let d = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(d, t);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_rotri_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui6,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let sh = ir.new_const(Type::I64, a.ui6 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_rotr(Type::I64, d, s, sh);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    // --- Count leading/trailing zeros/ones ---

    fn trans_clz_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        let zv = ir.new_const(Type::I64, 32);
        ir.gen_clz(Type::I32, d, s, zv);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_clz_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        let zv = ir.new_const(Type::I64, 64);
        ir.gen_clz(Type::I64, d, s, zv);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ctz_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        let zv = ir.new_const(Type::I64, 32);
        ir.gen_ctz(Type::I32, d, s, zv);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ctz_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        let zv = ir.new_const(Type::I64, 64);
        ir.gen_ctz(Type::I64, d, s, zv);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_clo_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let inv = ir.new_temp(Type::I64);
        ir.gen_not(Type::I64, inv, s);
        let d = ir.new_temp(Type::I64);
        let zv = ir.new_const(Type::I64, 32);
        ir.gen_clz(Type::I32, d, inv, zv);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_clo_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let inv = ir.new_temp(Type::I64);
        ir.gen_not(Type::I64, inv, s);
        let d = ir.new_temp(Type::I64);
        let zv = ir.new_const(Type::I64, 64);
        ir.gen_clz(Type::I64, d, inv, zv);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_cto_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let inv = ir.new_temp(Type::I64);
        ir.gen_not(Type::I64, inv, s);
        let d = ir.new_temp(Type::I64);
        let zv = ir.new_const(Type::I64, 32);
        ir.gen_ctz(Type::I32, d, inv, zv);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_cto_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let inv = ir.new_temp(Type::I64);
        ir.gen_not(Type::I64, inv, s);
        let d = ir.new_temp(Type::I64);
        let zv = ir.new_const(Type::I64, 64);
        ir.gen_ctz(Type::I64, d, inv, zv);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    // --- Byte reversal (via bswap) ---

    fn trans_revb_2h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_revb_2h as *const () as u64,
            &[s],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_revb_4h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_revb_4h as *const () as u64,
            &[s],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_revb_2w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_revb_2w as *const () as u64,
            &[s],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_revb_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_bswap64(Type::I64, d, s, 0);
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_revh_2w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_revh_2w as *const () as u64,
            &[s],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_revh_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_revh_d as *const () as u64,
            &[s],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bitrev_4b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_bitrev_4b as *const () as u64,
            &[s],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bitrev_8b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_bitrev_8b as *const () as u64,
            &[s],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bitrev_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_bitrev_w as *const () as u64,
            &[s],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_bitrev_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_bitrev_d as *const () as u64,
            &[s],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    // --- MULH (high multiply via helpers) ---

    fn trans_mulh_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_mulh_d as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mulh_du(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_mulh_du as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mulh_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_mulh_w as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_mulh_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_mulh_wu as *const () as u64,
            &[s1, s2],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    // --- Load/Store (Phase 2, task11) ---

    fn trans_ld_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let d = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, d, addr, u32::from(MemOp::sb().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ld_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let d = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, d, addr, u32::from(MemOp::sw().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ld_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let d = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, d, addr, u32::from(MemOp::sl().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ld_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let d = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, d, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ld_bu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let d = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, d, addr, u32::from(MemOp::ub().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ld_hu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let d = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, d, addr, u32::from(MemOp::uw().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ld_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let d = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, d, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_st_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        ir.gen_qemu_st(Type::I64, val, addr, u32::from(MemOp::ub().bits()));
        true
    }

    fn trans_st_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        ir.gen_qemu_st(Type::I64, val, addr, u32::from(MemOp::uw().bits()));
        true
    }

    fn trans_st_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        ir.gen_qemu_st(Type::I64, val, addr, u32::from(MemOp::ul().bits()));
        true
    }

    fn trans_st_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        ir.gen_qemu_st(Type::I64, val, addr, u32::from(MemOp::uq().bits()));
        true
    }

    fn trans_ldx_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sb(),
        );
        true
    }

    fn trans_ldx_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sw(),
        );
        true
    }

    fn trans_ldx_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sl(),
        );
        true
    }

    fn trans_ldx_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    fn trans_stx_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::ub(),
        );
        true
    }

    fn trans_stx_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uw(),
        );
        true
    }

    fn trans_stx_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
        );
        true
    }

    fn trans_stx_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    fn trans_ldx_bu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::ub(),
        );
        true
    }

    fn trans_ldx_hu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uw(),
        );
        true
    }

    fn trans_ldx_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_index_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
        );
        true
    }

    fn trans_preldx(
        &mut self,
        _ir: &mut Context,
        _a: &insn_decode::ArgsHintR3,
    ) -> bool {
        true
    }

    fn trans_preld(
        &mut self,
        _ir: &mut Context,
        _a: &insn_decode::ArgsHintRSi12,
    ) -> bool {
        true
    }

    fn trans_ldptr_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si14,
    ) -> bool {
        use gen_common::gpr_get;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, (a.si14 << 2) as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sl(),
        );
        true
    }

    fn trans_stptr_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si14,
    ) -> bool {
        use gen_common::gpr_get;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, (a.si14 << 2) as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
        );
        true
    }

    fn trans_ldptr_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si14,
    ) -> bool {
        use gen_common::gpr_get;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, (a.si14 << 2) as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    fn trans_stptr_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si14,
    ) -> bool {
        use gen_common::gpr_get;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, (a.si14 << 2) as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    fn trans_ldgt_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sb(),
        );
        true
    }

    fn trans_ldgt_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sw(),
        );
        true
    }

    fn trans_ldgt_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sl(),
        );
        true
    }

    fn trans_ldgt_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    fn trans_ldle_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sb(),
        );
        true
    }

    fn trans_ldle_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sw(),
        );
        true
    }

    fn trans_ldle_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::sl(),
        );
        true
    }

    fn trans_ldle_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        gen_gload_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    fn trans_stgt_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::ub(),
        );
        true
    }

    fn trans_stgt_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uw(),
        );
        true
    }

    fn trans_stgt_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
        );
        true
    }

    fn trans_stgt_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    fn trans_stle_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::ub(),
        );
        true
    }

    fn trans_stle_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uw(),
        );
        true
    }

    fn trans_stle_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
        );
        true
    }

    fn trans_stle_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        let addr = gen_predicate_addr(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        gen_gstore_addr(
            self,
            ir,
            a.rd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    // --- Branch/Jump (Phase 2, task12) ---

    fn trans_beq(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si16,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.si16 << 2) as u64);
        let next_pc = self.base.pc_next;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rd as u8);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, s1, s2, Cond::Eq, label_taken);
        let c_next = ir.new_const(Type::I64, next_pc);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_bne(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si16,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.si16 << 2) as u64);
        let next_pc = self.base.pc_next;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rd as u8);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, s1, s2, Cond::Ne, label_taken);
        let c_next = ir.new_const(Type::I64, next_pc);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_blt(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si16,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.si16 << 2) as u64);
        let next_pc = self.base.pc_next;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rd as u8);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, s1, s2, Cond::Lt, label_taken);
        let c_next = ir.new_const(Type::I64, next_pc);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_bge(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si16,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.si16 << 2) as u64);
        let next_pc = self.base.pc_next;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rd as u8);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, s1, s2, Cond::Ge, label_taken);
        let c_next = ir.new_const(Type::I64, next_pc);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_bltu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si16,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.si16 << 2) as u64);
        let next_pc = self.base.pc_next;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rd as u8);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, s1, s2, Cond::Ltu, label_taken);
        let c_next = ir.new_const(Type::I64, next_pc);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_bgeu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si16,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.si16 << 2) as u64);
        let next_pc = self.base.pc_next;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let s2 = gpr_get(&self.gpr, ir, a.rd as u8);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, s1, s2, Cond::Geu, label_taken);
        let c_next = ir.new_const(Type::I64, next_pc);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_beqz(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR1Offs21,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.offs21 << 2) as u64);
        let next_pc = self.base.pc_next;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let zero = ir.new_const(Type::I64, 0);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, s1, zero, Cond::Eq, label_taken);
        let c_next = ir.new_const(Type::I64, next_pc);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_bnez(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR1Offs21,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.offs21 << 2) as u64);
        let next_pc = self.base.pc_next;
        let s1 = gpr_get(&self.gpr, ir, a.rj as u8);
        let zero = ir.new_const(Type::I64, 0);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, s1, zero, Cond::Ne, label_taken);
        let c_next = ir.new_const(Type::I64, next_pc);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsOffs26,
    ) -> bool {
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.offs26 << 2) as u64);
        let c = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_bl(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsOffs26,
    ) -> bool {
        use gen_common::gpr_set;
        let pc = self.base.pc_next - 4;
        let next_pc = self.base.pc_next;
        let target = pc.wrapping_add((a.offs26 << 2) as u64);
        let ret = ir.new_const(Type::I64, next_pc);
        gpr_set(&self.gpr, ir, 1, ret);
        let c = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_jirl(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si16,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::tb::TB_EXIT_NOCHAIN;
        let next_pc = self.base.pc_next;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, (a.si16 << 2) as u64);
        let target = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, target, base, off);
        let ret = ir.new_const(Type::I64, next_pc);
        gpr_set(&self.gpr, ir, a.rd as u8, ret);
        ir.gen_mov(Type::I64, self.pc, target);
        ir.gen_exit_tb(TB_EXIT_NOCHAIN);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_dbar(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsCode,
    ) -> bool {
        ir.gen_mb(0);
        true
    }

    fn trans_ibar(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsCode,
    ) -> bool {
        let d = ir.new_temp(Type::I64);
        ir.gen_mb(0);
        ir.gen_call(
            d,
            helpers::loongarch_helper_ibar as *const () as u64,
            &[self.env],
        );
        self.base.is_jmp = DisasJumpType::TooMany;
        true
    }

    fn trans_ll_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si14,
    ) -> bool {
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, (a.si14 << 2) as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let d = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, d, addr, u32::from(MemOp::sl().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        let one = ir.new_const(Type::I64, 1);
        ir.gen_mov(Type::I64, self.llbctl, one);
        ir.gen_mov(Type::I64, self.ll_res_addr, addr);
        ir.gen_mov(Type::I64, self.ll_res_val, d);
        true
    }

    fn trans_ll_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si14,
    ) -> bool {
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, (a.si14 << 2) as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let d = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, d, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        let one = ir.new_const(Type::I64, 1);
        ir.gen_mov(Type::I64, self.llbctl, one);
        ir.gen_mov(Type::I64, self.ll_res_addr, addr);
        ir.gen_mov(Type::I64, self.ll_res_val, d);
        true
    }

    fn trans_sc_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si14,
    ) -> bool {
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, (a.si14 << 2) as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        let old_rd = gpr_get(&self.gpr, ir, a.rd as u8);
        let env_tmp = self.env;
        let status = ir.new_temp(Type::I64);
        ir.gen_call(
            status,
            helpers::loongarch_helper_sc_w as *const () as u64,
            &[env_tmp, addr, val],
        );
        let trap_status = ir.new_const(Type::I64, 2);
        let rd_val = ir.new_temp(Type::I64);
        ir.gen_movcond(
            Type::I64,
            rd_val,
            status,
            trap_status,
            old_rd,
            status,
            Cond::Geu,
        );
        gpr_set(&self.gpr, ir, a.rd as u8, rd_val);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, status, trap_status, Cond::Ltu, label_ok);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        true
    }

    fn trans_sc_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si14,
    ) -> bool {
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, (a.si14 << 2) as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        let old_rd = gpr_get(&self.gpr, ir, a.rd as u8);
        let env_tmp = self.env;
        let status = ir.new_temp(Type::I64);
        ir.gen_call(
            status,
            helpers::loongarch_helper_sc_d as *const () as u64,
            &[env_tmp, addr, val],
        );
        let trap_status = ir.new_const(Type::I64, 2);
        let rd_val = ir.new_temp(Type::I64);
        ir.gen_movcond(
            Type::I64,
            rd_val,
            status,
            trap_status,
            old_rd,
            status,
            Cond::Geu,
        );
        gpr_set(&self.gpr, ir, a.rd as u8, rd_val);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, status, trap_status, Cond::Ltu, label_ok);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        true
    }

    fn trans_amadd_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::uq().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, new, old, src);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amadd_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::sl().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, new, old, src);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amswap_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::uq().bits()));
        ir.gen_qemu_st(Type::I64, src, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amswap_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::sl().bits()));
        ir.gen_qemu_st(Type::I64, src, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amand_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::uq().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, new, old, src);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amand_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::sl().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, new, old, src);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amor_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::uq().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, new, old, src);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amor_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::sl().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_or(Type::I64, new, old, src);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amxor_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::uq().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_xor(Type::I64, new, old, src);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amxor_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::MemOp;
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::sl().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_xor(Type::I64, new, old, src);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_ammax_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::{Cond, MemOp};
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::uq().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, new, old, src, old, src, Cond::Ge);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_ammin_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::{Cond, MemOp};
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::uq().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, new, old, src, old, src, Cond::Le);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_ammax_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::{Cond, MemOp};
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::sl().bits()));
        let src_ext = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(src_ext, src);
        let new = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, new, old, src_ext, old, src_ext, Cond::Ge);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_ammin_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::{Cond, MemOp};
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::sl().bits()));
        let src_ext = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(src_ext, src);
        let new = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, new, old, src_ext, old, src_ext, Cond::Le);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_ammax_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::{Cond, MemOp};
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::sl().bits()));
        let old_u = ir.new_temp(Type::I64);
        let mask32 = ir.new_const(Type::I64, 0xFFFF_FFFF);
        ir.gen_and(Type::I64, old_u, old, mask32);
        let src_u = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, src_u, src, mask32);
        let new = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, new, old_u, src_u, old_u, src_u, Cond::Geu);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_ammin_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::{Cond, MemOp};
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::sl().bits()));
        let old_u = ir.new_temp(Type::I64);
        let mask32 = ir.new_const(Type::I64, 0xFFFF_FFFF);
        ir.gen_and(Type::I64, old_u, old, mask32);
        let src_u = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, src_u, src, mask32);
        let new = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, new, old_u, src_u, old_u, src_u, Cond::Leu);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::ul().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_ammax_du(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::{Cond, MemOp};
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::uq().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, new, old, src, old, src, Cond::Geu);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_ammin_du(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        if am_r3_has_forbidden_overlap(a) {
            return false;
        }
        ir.contains_atomic = true;
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::{Cond, MemOp};
        let addr = gpr_get(&self.gpr, ir, a.rj as u8);
        let src = gpr_get(&self.gpr, ir, a.rk as u8);
        let old = ir.new_temp(Type::I64);
        ir.gen_qemu_ld(Type::I64, old, addr, u32::from(MemOp::uq().bits()));
        let new = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, new, old, src, old, src, Cond::Leu);
        ir.gen_qemu_st(Type::I64, new, addr, u32::from(MemOp::uq().bits()));
        gpr_set(&self.gpr, ir, a.rd as u8, old);
        true
    }

    fn trans_amswap_db_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amswap_w(ir, a)
    }

    fn trans_amswap_db_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amswap_d(ir, a)
    }

    fn trans_amadd_db_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amadd_w(ir, a)
    }

    fn trans_amadd_db_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amadd_d(ir, a)
    }

    fn trans_amand_db_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amand_w(ir, a)
    }

    fn trans_amand_db_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amand_d(ir, a)
    }

    fn trans_amor_db_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amor_w(ir, a)
    }

    fn trans_amor_db_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amor_d(ir, a)
    }

    fn trans_amxor_db_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amxor_w(ir, a)
    }

    fn trans_amxor_db_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_amxor_d(ir, a)
    }

    fn trans_ammax_db_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_ammax_w(ir, a)
    }

    fn trans_ammax_db_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_ammax_d(ir, a)
    }

    fn trans_ammin_db_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_ammin_w(ir, a)
    }

    fn trans_ammin_db_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_ammin_d(ir, a)
    }

    fn trans_ammax_db_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_ammax_wu(ir, a)
    }

    fn trans_ammax_db_du(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_ammax_du(ir, a)
    }

    fn trans_ammin_db_wu(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_ammin_wu(ir, a)
    }

    fn trans_ammin_db_du(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        self.trans_ammin_du(ir, a)
    }

    fn trans_rdtimel_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_rdtime(self, ir, a.rd as u8, a.rj as u8, true, false);
        true
    }

    fn trans_rdtimeh_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_rdtime(self, ir, a.rd as u8, a.rj as u8, true, true);
        true
    }

    fn trans_rdtime_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_rdtime(self, ir, a.rd as u8, a.rj as u8, false, false);
        true
    }

    fn trans_csrrd(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsCsr,
    ) -> bool {
        use gen_common::gpr_set;
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        gen_guest_csr_check(self, ir, a.csr_num as u32);
        let csr_num = ir.new_const(Type::I64, a.csr_num as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_csrrd as *const () as u64,
            &[env_tmp, csr_num],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_csrwr(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsCsr,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        gen_guest_csr_check(self, ir, a.csr_num as u32);
        let csr_num = ir.new_const(Type::I64, a.csr_num as u64);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_csrwr as *const () as u64,
            &[env_tmp, csr_num, val],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_csrxchg(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsCsr,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        if a.rj == 0 || a.rj == 1 {
            return false;
        }
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        gen_guest_csr_check(self, ir, a.csr_num as u32);
        let csr_num = ir.new_const(Type::I64, a.csr_num as u64);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        let mask = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_csrxchg as *const () as u64,
            &[env_tmp, csr_num, val, mask],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_gcsrrd(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsCsr,
    ) -> bool {
        use gen_common::gpr_set;
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let csr_num = ir.new_const(Type::I64, a.csr_num as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_gcsrrd as *const () as u64,
            &[env_tmp, csr_num],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_gcsrwr(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsCsr,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let csr_num = ir.new_const(Type::I64, a.csr_num as u64);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_gcsrwr as *const () as u64,
            &[env_tmp, csr_num, val],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_gcsrxchg(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsCsr,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        if a.rj == 0 || a.rj == 1 {
            return false;
        }
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let csr_num = ir.new_const(Type::I64, a.csr_num as u64);
        let val = gpr_get(&self.gpr, ir, a.rd as u8);
        let mask = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_gcsrxchg as *const () as u64,
            &[env_tmp, csr_num, val, mask],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_cpucfg(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        gen_guest_sensitive_check(self, ir);
        let idx = gpr_get(&self.gpr, ir, a.rj as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_cpucfg as *const () as u64,
            &[env_tmp, idx],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_syscall(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsCode,
    ) -> bool {
        let env_tmp = self.env;
        let pc = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc);
        let ecode = ir.new_const(Type::I64, u64::from(ECODE_SYS));
        let esubcode = ir.new_const(Type::I64, 0);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_raise_exception as *const () as u64,
            &[env_tmp, ecode, esubcode],
        );
        ir.gen_mov(Type::I64, self.pc, d);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_break_(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsCode,
    ) -> bool {
        let env_tmp = self.env;
        let pc = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc);
        let ecode = ir.new_const(Type::I64, u64::from(ECODE_BRK));
        let esubcode = ir.new_const(Type::I64, 0);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_raise_exception as *const () as u64,
            &[env_tmp, ecode, esubcode],
        );
        ir.gen_mov(Type::I64, self.pc, d);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_hvcl(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsCode,
    ) -> bool {
        let env_tmp = self.env;
        let pc = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc);
        let ecode = ir.new_const(Type::I64, u64::from(ECODE_HVC));
        let esubcode = ir.new_const(Type::I64, 0);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_raise_exception as *const () as u64,
            &[env_tmp, ecode, esubcode],
        );
        ir.gen_mov(Type::I64, self.pc, d);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_ertn(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsEmpty,
    ) -> bool {
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_ertn as *const () as u64,
            &[env_tmp],
        );
        ir.gen_mov(Type::I64, self.pc, d);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_idle(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsCode,
    ) -> bool {
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        gen_guest_sensitive_check(self, ir);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_idle as *const () as u64,
            &[env_tmp],
        );
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_exit_tb(machina_accel::ir::tb::EXCP_LOONGARCH_WFI);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_tlbsrch(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsEmpty,
    ) -> bool {
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_tlbsrch as *const () as u64,
            &[env_tmp],
        );
        true
    }

    fn trans_tlbrd(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsEmpty,
    ) -> bool {
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_tlbrd as *const () as u64,
            &[env_tmp],
        );
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_tlbwr(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsEmpty,
    ) -> bool {
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_tlbwr as *const () as u64,
            &[env_tmp],
        );
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_tlbfill(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsEmpty,
    ) -> bool {
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_tlbfill as *const () as u64,
            &[env_tmp],
        );
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_tlbclr(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsEmpty,
    ) -> bool {
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_tlbclr as *const () as u64,
            &[env_tmp],
        );
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_tlbflush(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsEmpty,
    ) -> bool {
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_tlbflush as *const () as u64,
            &[env_tmp],
        );
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_cacop(
        &mut self,
        ir: &mut Context,
        _a: &insn_decode::ArgsCopRSi12,
    ) -> bool {
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        gen_guest_sensitive_check(self, ir);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        true
    }

    fn trans_lddir(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui8,
    ) -> bool {
        use gen_common::{gpr_get, gpr_set};
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let level = ir.new_const(Type::I64, a.ui8 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_lddir as *const () as u64,
            &[env_tmp, base, level],
        );
        gpr_set(&self.gpr, ir, a.rd as u8, d);
        true
    }

    fn trans_ldpte(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Ui8,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let odd = ir.new_const(Type::I64, a.ui8 as u64);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_ldpte as *const () as u64,
            &[env_tmp, base, odd],
        );
        true
    }

    fn trans_invtlb(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR3,
    ) -> bool {
        use gen_common::gpr_get;
        use machina_accel::ir::Cond;
        let env_tmp = self.env;
        let pc_val = ir.new_const(Type::I64, self.base.pc_next - 4);
        ir.gen_mov(Type::I64, self.pc, pc_val);
        let chk = ir.new_temp(Type::I64);
        ir.gen_call(
            chk,
            helpers::loongarch_helper_check_plv as *const () as u64,
            &[env_tmp],
        );
        let zero = ir.new_const(Type::I64, 0);
        let label_ok = ir.new_label();
        ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
        ir.gen_mov(Type::I64, self.pc, chk);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_ok);
        let op = ir.new_const(Type::I64, a.rd as u64);
        let asid_val = gpr_get(&self.gpr, ir, a.rj as u8);
        let va = gpr_get(&self.gpr, ir, a.rk as u8);
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_invtlb as *const () as u64,
            &[env_tmp, op, asid_val, va],
        );
        let zero2 = ir.new_const(Type::I64, 0);
        let label_done = ir.new_label();
        ir.gen_brcond(Type::I64, d, zero2, Cond::Eq, label_done);
        ir.gen_mov(Type::I64, self.pc, d);
        ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
        ir.gen_set_label(label_done);
        let pc_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, pc_next);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_iocsrrd_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_iocsr_rd(self, ir, a, 1)
    }

    fn trans_iocsrrd_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_iocsr_rd(self, ir, a, 2)
    }

    fn trans_iocsrrd_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_iocsr_rd(self, ir, a, 4)
    }

    fn trans_iocsrrd_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_iocsr_rd(self, ir, a, 8)
    }

    fn trans_iocsrwr_b(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_iocsr_wr(self, ir, a, 1)
    }

    fn trans_iocsrwr_h(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_iocsr_wr(self, ir, a, 2)
    }

    fn trans_iocsrwr_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_iocsr_wr(self, ir, a, 4)
    }

    fn trans_iocsrwr_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2,
    ) -> bool {
        gen_iocsr_wr(self, ir, a, 8)
    }

    // FP load/store
    fn trans_fld_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_get;
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        gen_fload_addr(self, ir, a.rd as u8, addr, MemOp::ul(), true);
        true
    }

    fn trans_fld_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_get;
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        gen_fload_addr(self, ir, a.rd as u8, addr, MemOp::uq(), false);
        true
    }

    fn trans_fst_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_get;
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        gen_fstore_addr(self, ir, a.rd as u8, addr, MemOp::ul());
        true
    }

    fn trans_fst_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsR2Si12,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_get;
        use machina_accel::ir::MemOp;
        let base = gpr_get(&self.gpr, ir, a.rj as u8);
        let off = ir.new_const(Type::I64, a.si12 as u64);
        let addr = ir.new_temp(Type::I64);
        ir.gen_add(Type::I64, addr, base, off);
        gen_fstore_addr(self, ir, a.rd as u8, addr, MemOp::uq());
        true
    }

    fn trans_fldx_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fload_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
            true,
        );
        true
    }

    fn trans_fldx_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fload_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
            false,
        );
        true
    }

    fn trans_fstx_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fstore_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
        );
        true
    }

    fn trans_fstx_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fstore_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    fn trans_fldgt_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_fp_predicate_assert(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fload_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
            true,
        );
        true
    }

    fn trans_fldgt_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_fp_predicate_assert(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fload_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
            false,
        );
        true
    }

    fn trans_fldle_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_fp_predicate_assert(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fload_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
            true,
        );
        true
    }

    fn trans_fldle_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_fp_predicate_assert(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fload_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
            false,
        );
        true
    }

    fn trans_fstgt_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_fp_predicate_assert(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fstore_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
        );
        true
    }

    fn trans_fstgt_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_fp_predicate_assert(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtgt_d,
        );
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fstore_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    fn trans_fstle_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_fp_predicate_assert(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fstore_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::ul(),
        );
        true
    }

    fn trans_fstle_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFrr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_fp_predicate_assert(
            self,
            ir,
            a.rj as u8,
            a.rk as u8,
            helpers::loongarch_helper_asrtle_d,
        );
        let addr = gen_fp_addr(self, ir, a.rj as u8, a.rk as u8);
        gen_fstore_addr(
            self,
            ir,
            a.fd as u8,
            addr,
            machina_accel::ir::MemOp::uq(),
        );
        true
    }

    // FP arithmetic
    fn trans_fadd_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fadd_s_fcsr,
        );
        true
    }

    fn trans_fadd_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fadd_d_fcsr,
        );
        true
    }

    fn trans_fsub_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fsub_s_fcsr,
        );
        true
    }

    fn trans_fsub_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fsub_d_fcsr,
        );
        true
    }

    fn trans_fmul_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmul_s_fcsr,
        );
        true
    }

    fn trans_fmul_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmul_d_fcsr,
        );
        true
    }

    fn trans_fdiv_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fdiv_s_fcsr,
        );
        true
    }

    fn trans_fdiv_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fdiv_d_fcsr,
        );
        true
    }

    fn trans_fmax_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmax_s_fcsr,
        );
        true
    }

    fn trans_fmax_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmax_d_fcsr,
        );
        true
    }

    fn trans_fmin_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmin_s_fcsr,
        );
        true
    }

    fn trans_fmin_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmin_d_fcsr,
        );
        true
    }

    fn trans_fmaxa_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmaxa_s_fcsr,
        );
        true
    }

    fn trans_fmaxa_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmaxa_d_fcsr,
        );
        true
    }

    fn trans_fmina_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmina_s_fcsr,
        );
        true
    }

    fn trans_fmina_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fmina_d_fcsr,
        );
        true
    }

    fn trans_fscaleb_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fscaleb_s_fcsr,
        );
        true
    }

    fn trans_fscaleb_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fscaleb_d_fcsr,
        );
        true
    }

    fn trans_fcopysign_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcopysign_s_fcsr,
        );
        true
    }

    fn trans_fcopysign_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr3,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_arith_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcopysign_d_fcsr,
        );
        true
    }

    fn trans_fsqrt_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_fsqrt_s_fcsr,
        );
        true
    }

    fn trans_fsqrt_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_fsqrt_d_fcsr,
        );
        true
    }

    fn trans_frecip_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_frecip_s_fcsr,
        );
        true
    }

    fn trans_frecip_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_frecip_d_fcsr,
        );
        true
    }

    fn trans_frsqrt_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_frsqrt_s_fcsr,
        );
        true
    }

    fn trans_frsqrt_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_frsqrt_d_fcsr,
        );
        true
    }

    fn trans_frecipe_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        self.trans_frecip_s(ir, a)
    }

    fn trans_frecipe_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        self.trans_frecip_d(ir, a)
    }

    fn trans_frsqrte_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        self.trans_frsqrt_s(ir, a)
    }

    fn trans_frsqrte_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        self.trans_frsqrt_d(ir, a)
    }

    fn trans_flogb_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_flogb_s_fcsr,
        );
        true
    }

    fn trans_flogb_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_flogb_d_fcsr,
        );
        true
    }

    fn trans_fclass_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_fclass_s_fcsr,
        );
        true
    }

    fn trans_fclass_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_fclass_d_fcsr,
        );
        true
    }

    fn trans_frint_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_frint_s_fcsr,
        );
        true
    }

    fn trans_frint_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_frint_d_fcsr,
        );
        true
    }

    // FP conversion
    fn trans_fcvt_s_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_fcvt_s_d_fcsr,
        );
        true
    }

    fn trans_fcvt_d_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_fcvt_d_s_fcsr,
        );
        true
    }

    fn trans_ffint_s_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ffint_s_w_fcsr,
        );
        true
    }

    fn trans_ffint_d_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ffint_d_w_fcsr,
        );
        true
    }

    fn trans_ffint_s_l(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ffint_s_l_fcsr,
        );
        true
    }

    fn trans_ffint_d_l(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ffint_d_l_fcsr,
        );
        true
    }

    fn trans_ftintrm_w_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrm_w_s_fcsr,
        );
        true
    }

    fn trans_ftintrm_w_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrm_w_d_fcsr,
        );
        true
    }

    fn trans_ftintrm_l_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrm_l_s_fcsr,
        );
        true
    }

    fn trans_ftintrm_l_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrm_l_d_fcsr,
        );
        true
    }

    fn trans_ftintrp_w_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrp_w_s_fcsr,
        );
        true
    }

    fn trans_ftintrp_w_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrp_w_d_fcsr,
        );
        true
    }

    fn trans_ftintrp_l_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrp_l_s_fcsr,
        );
        true
    }

    fn trans_ftintrp_l_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrp_l_d_fcsr,
        );
        true
    }

    fn trans_ftintrz_w_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrz_w_s_fcsr,
        );
        true
    }

    fn trans_ftintrz_w_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrz_w_d_fcsr,
        );
        true
    }

    fn trans_ftintrz_l_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrz_l_s_fcsr,
        );
        true
    }

    fn trans_ftintrz_l_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrz_l_d_fcsr,
        );
        true
    }

    fn trans_ftintrne_w_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrne_w_s_fcsr,
        );
        true
    }

    fn trans_ftintrne_w_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrne_w_d_fcsr,
        );
        true
    }

    fn trans_ftintrne_l_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrne_l_s_fcsr,
        );
        true
    }

    fn trans_ftintrne_l_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftintrne_l_d_fcsr,
        );
        true
    }

    fn trans_ftint_w_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftint_w_s_fcsr,
        );
        true
    }

    fn trans_ftint_w_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftint_w_d_fcsr,
        );
        true
    }

    fn trans_ftint_l_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftint_l_s_fcsr,
        );
        true
    }

    fn trans_ftint_l_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_unary_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            helpers::loongarch_helper_ftint_l_d_fcsr,
        );
        true
    }

    // FP compare (subset: CEQ, CLT, CLE, CUN)
    fn trans_fcmp_ceq_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_ceq_s_fcsr,
        );
        true
    }

    fn trans_fcmp_ceq_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_ceq_d_fcsr,
        );
        true
    }

    fn trans_fcmp_clt_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_clt_s_fcsr,
        );
        true
    }

    fn trans_fcmp_clt_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_clt_d_fcsr,
        );
        true
    }

    fn trans_fcmp_cle_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cle_s_fcsr,
        );
        true
    }

    fn trans_fcmp_cle_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cle_d_fcsr,
        );
        true
    }

    fn trans_fcmp_cun_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cun_s_fcsr,
        );
        true
    }

    fn trans_fcmp_cun_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cun_d_fcsr,
        );
        true
    }

    // FP move
    fn trans_fmov_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        let boxed = gen_nanbox_s_value(ir, v);
        gen_float::fpr_set(self, ir, a.fd as u8, boxed);
        true
    }

    fn trans_fmov_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        gen_float::fpr_set(self, ir, a.fd as u8, v);
        true
    }

    fn trans_movgr2fr_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_get;
        let v = gpr_get(&self.gpr, ir, a.fj as u8);
        let old = gen_float::fpr_get(self, ir, a.fd as u8);
        let clear_high = ir.new_const(Type::I64, 0xffff_ffff_0000_0000);
        let low_mask = ir.new_const(Type::I64, 0xffff_ffff);
        let old_high = ir.new_temp(Type::I64);
        let v_low = ir.new_temp(Type::I64);
        let d = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, old_high, old, clear_high);
        ir.gen_and(Type::I64, v_low, v, low_mask);
        ir.gen_or(Type::I64, d, old_high, v_low);
        gen_float::fpr_set(self, ir, a.fd as u8, d);
        true
    }

    fn trans_movgr2fr_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_get;
        let v = gpr_get(&self.gpr, ir, a.fj as u8);
        gen_float::fpr_set(self, ir, a.fd as u8, v);
        true
    }

    fn trans_movgr2frh_w(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_get;
        let v = gpr_get(&self.gpr, ir, a.fj as u8);
        let old = gen_float::fpr_get(self, ir, a.fd as u8);
        let low_mask = ir.new_const(Type::I64, 0xffff_ffff);
        let shift = ir.new_const(Type::I64, 32);
        let old_low = ir.new_temp(Type::I64);
        let v_low = ir.new_temp(Type::I64);
        let v_high = ir.new_temp(Type::I64);
        let d = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, old_low, old, low_mask);
        ir.gen_and(Type::I64, v_low, v, low_mask);
        ir.gen_shl(Type::I64, v_high, v_low, shift);
        ir.gen_or(Type::I64, d, old_low, v_high);
        gen_float::fpr_set(self, ir, a.fd as u8, d);
        true
    }

    fn trans_movfr2gr_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_set;
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        let ext = ir.new_temp(Type::I64);
        ir.gen_ext_i32_i64(ext, v);
        gpr_set(&self.gpr, ir, a.fd as u8, ext);
        true
    }

    fn trans_movfrh2gr_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_set;
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        let shift = ir.new_const(Type::I64, 32);
        let high = ir.new_temp(Type::I64);
        let ext = ir.new_temp(Type::I64);
        ir.gen_shr(Type::I64, high, v, shift);
        ir.gen_ext_i32_i64(ext, high);
        gpr_set(&self.gpr, ir, a.fd as u8, ext);
        true
    }

    fn trans_movfr2gr_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_set;
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        gpr_set(&self.gpr, ir, a.fd as u8, v);
        true
    }

    fn trans_movgr2fcsr(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_get;
        let env_tmp = self.env;
        let val = gpr_get(&self.gpr, ir, a.fj as u8);
        let fcsrd = ir.new_const(Type::I64, u64::from(a.fd as u8));
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_movgr2fcsr_idx as *const () as u64,
            &[env_tmp, val, fcsrd],
        );
        true
    }

    fn trans_movfcsr2gr(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_set;
        let env_tmp = self.env;
        let fcsrs = ir.new_const(Type::I64, u64::from(a.fj as u8));
        let d = ir.new_temp(Type::I64);
        ir.gen_call(
            d,
            helpers::loongarch_helper_movfcsr2gr_idx as *const () as u64,
            &[env_tmp, fcsrs],
        );
        gpr_set(&self.gpr, ir, a.fd as u8, d);
        true
    }

    fn trans_movfr2cf(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsCf,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        let one = ir.new_const(Type::I64, 1);
        let bit = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, bit, v, one);
        gen_float::fcc_set(self, ir, a.cd as u8, bit);
        true
    }

    fn trans_movcf2fr(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFc,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let v = gen_float::fcc_get(self, ir, a.cj as u8);
        gen_float::fpr_set(self, ir, a.fd as u8, v);
        true
    }

    fn trans_movgr2cf(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsCr,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_get;
        let v = gpr_get(&self.gpr, ir, a.rj as u8);
        let one = ir.new_const(Type::I64, 1);
        let bit = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, bit, v, one);
        gen_float::fcc_set(self, ir, a.cd as u8, bit);
        true
    }

    fn trans_movcf2gr(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsRc,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use gen_common::gpr_set;
        let v = gen_float::fcc_get(self, ir, a.cj as u8);
        gpr_set(&self.gpr, ir, a.rd as u8, v);
        true
    }

    fn trans_fabs_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        let mask = ir.new_const(Type::I64, 0x7FFF_FFFF);
        let d = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, d, v, mask);
        let boxed = gen_nanbox_s_value(ir, d);
        gen_float::fpr_set(self, ir, a.fd as u8, boxed);
        true
    }

    fn trans_fabs_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        let mask = ir.new_const(Type::I64, 0x7FFF_FFFF_FFFF_FFFF);
        let d = ir.new_temp(Type::I64);
        ir.gen_and(Type::I64, d, v, mask);
        gen_float::fpr_set(self, ir, a.fd as u8, d);
        true
    }

    fn trans_fneg_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        let bit = ir.new_const(Type::I64, 0x8000_0000);
        let d = ir.new_temp(Type::I64);
        ir.gen_xor(Type::I64, d, v, bit);
        let boxed = gen_nanbox_s_value(ir, d);
        gen_float::fpr_set(self, ir, a.fd as u8, boxed);
        true
    }

    fn trans_fneg_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr2,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        let v = gen_float::fpr_get(self, ir, a.fj as u8);
        let bit = ir.new_const(Type::I64, 0x8000_0000_0000_0000);
        let d = ir.new_temp(Type::I64);
        ir.gen_xor(Type::I64, d, v, bit);
        gen_float::fpr_set(self, ir, a.fd as u8, d);
        true
    }

    fn trans_fmadd_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr4,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_fused_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            a.fa as u8,
            helpers::loongarch_helper_fmadd_s_fcsr,
        );
        true
    }

    fn trans_fmadd_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr4,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_fused_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            a.fa as u8,
            helpers::loongarch_helper_fmadd_d_fcsr,
        );
        true
    }

    fn trans_fmsub_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr4,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_fused_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            a.fa as u8,
            helpers::loongarch_helper_fmsub_s_fcsr,
        );
        true
    }

    fn trans_fmsub_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr4,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_fused_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            a.fa as u8,
            helpers::loongarch_helper_fmsub_d_fcsr,
        );
        true
    }

    fn trans_fnmadd_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr4,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_fused_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            a.fa as u8,
            helpers::loongarch_helper_fnmadd_s_fcsr,
        );
        true
    }

    fn trans_fnmadd_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr4,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_fused_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            a.fa as u8,
            helpers::loongarch_helper_fnmadd_d_fcsr,
        );
        true
    }

    fn trans_fnmsub_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr4,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_fused_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            a.fa as u8,
            helpers::loongarch_helper_fnmsub_s_fcsr,
        );
        true
    }

    fn trans_fnmsub_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFr4,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_fused_fcsr(
            self,
            ir,
            a.fd as u8,
            a.fj as u8,
            a.fk as u8,
            a.fa as u8,
            helpers::loongarch_helper_fnmsub_d_fcsr,
        );
        true
    }

    fn trans_fsel(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFsel,
    ) -> bool {
        use machina_accel::ir::Cond;
        gen_float::check_fpe(self, ir);
        let fj = gen_float::fpr_get(self, ir, a.fj as u8);
        let fk = gen_float::fpr_get(self, ir, a.fk as u8);
        let ca = gen_float::fcc_get(self, ir, a.ca as u8);
        let zero = ir.new_const(Type::I64, 0);
        let d = ir.new_temp(Type::I64);
        ir.gen_movcond(Type::I64, d, ca, zero, fk, fj, Cond::Ne);
        gen_float::fpr_set(self, ir, a.fd as u8, d);
        true
    }

    fn trans_bceqz(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFbranch,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let ca = gen_float::fcc_get(self, ir, a.cj as u8);
        let zero = ir.new_const(Type::I64, 0);
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.offs21 << 2) as u64);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, ca, zero, Cond::Eq, label_taken);
        let c_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    fn trans_bcnez(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFbranch,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        use machina_accel::ir::tb::TB_EXIT_IDX1;
        use machina_accel::ir::Cond;
        let ca = gen_float::fcc_get(self, ir, a.cj as u8);
        let zero = ir.new_const(Type::I64, 0);
        let pc = self.base.pc_next - 4;
        let target = pc.wrapping_add((a.offs21 << 2) as u64);
        let label_taken = ir.new_label();
        ir.gen_brcond(Type::I64, ca, zero, Cond::Ne, label_taken);
        let c_next = ir.new_const(Type::I64, self.base.pc_next);
        ir.gen_mov(Type::I64, self.pc, c_next);
        ir.gen_goto_tb(1);
        ir.gen_exit_tb(TB_EXIT_IDX1);
        ir.gen_set_label(label_taken);
        let c_target = ir.new_const(Type::I64, target);
        ir.gen_mov(Type::I64, self.pc, c_target);
        ir.gen_goto_tb(0);
        ir.gen_exit_tb(TB_EXIT_IDX0);
        self.base.is_jmp = DisasJumpType::NoReturn;
        true
    }

    // Remaining FCMP conditions delegate to existing helpers
    fn trans_fcmp_caf_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_caf_s_fcsr,
        );
        true
    }
    fn trans_fcmp_caf_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_caf_d_fcsr,
        );
        true
    }
    fn trans_fcmp_cueq_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cueq_s_fcsr,
        );
        true
    }
    fn trans_fcmp_cueq_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cueq_d_fcsr,
        );
        true
    }
    fn trans_fcmp_cult_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cult_s_fcsr,
        );
        true
    }
    fn trans_fcmp_cult_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cult_d_fcsr,
        );
        true
    }
    fn trans_fcmp_cule_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cule_s_fcsr,
        );
        true
    }
    fn trans_fcmp_cule_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cule_d_fcsr,
        );
        true
    }
    fn trans_fcmp_cne_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cne_s_fcsr,
        );
        true
    }
    fn trans_fcmp_cne_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cne_d_fcsr,
        );
        true
    }
    fn trans_fcmp_cor_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cor_s_fcsr,
        );
        true
    }
    fn trans_fcmp_cor_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cor_d_fcsr,
        );
        true
    }
    fn trans_fcmp_cune_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cune_s_fcsr,
        );
        true
    }
    fn trans_fcmp_cune_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_cune_d_fcsr,
        );
        true
    }
    // Signaling variants use the same predicates with stricter NaN invalid flags.
    fn trans_fcmp_saf_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_saf_s_fcsr,
        );
        true
    }
    fn trans_fcmp_saf_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_saf_d_fcsr,
        );
        true
    }
    fn trans_fcmp_seq_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_seq_s_fcsr,
        );
        true
    }
    fn trans_fcmp_seq_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_seq_d_fcsr,
        );
        true
    }
    fn trans_fcmp_slt_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_slt_s_fcsr,
        );
        true
    }
    fn trans_fcmp_slt_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_slt_d_fcsr,
        );
        true
    }
    fn trans_fcmp_sle_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sle_s_fcsr,
        );
        true
    }
    fn trans_fcmp_sle_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sle_d_fcsr,
        );
        true
    }
    fn trans_fcmp_sun_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sun_s_fcsr,
        );
        true
    }
    fn trans_fcmp_sun_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sun_d_fcsr,
        );
        true
    }
    fn trans_fcmp_sueq_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sueq_s_fcsr,
        );
        true
    }
    fn trans_fcmp_sueq_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sueq_d_fcsr,
        );
        true
    }
    fn trans_fcmp_sult_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sult_s_fcsr,
        );
        true
    }
    fn trans_fcmp_sult_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sult_d_fcsr,
        );
        true
    }
    fn trans_fcmp_sule_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sule_s_fcsr,
        );
        true
    }
    fn trans_fcmp_sule_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sule_d_fcsr,
        );
        true
    }
    fn trans_fcmp_sne_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sne_s_fcsr,
        );
        true
    }
    fn trans_fcmp_sne_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sne_d_fcsr,
        );
        true
    }
    fn trans_fcmp_sor_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sor_s_fcsr,
        );
        true
    }
    fn trans_fcmp_sor_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sor_d_fcsr,
        );
        true
    }
    fn trans_fcmp_sune_s(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sune_s_fcsr,
        );
        true
    }
    fn trans_fcmp_sune_d(
        &mut self,
        ir: &mut Context,
        a: &insn_decode::ArgsFcmp,
    ) -> bool {
        gen_float::check_fpe(self, ir);
        gen_float::gen_fp_cmp_fcsr(
            self,
            ir,
            a.cd as u8,
            a.fj as u8,
            a.fk as u8,
            helpers::loongarch_helper_fcmp_sune_d_fcsr,
        );
        true
    }
}

#[allow(clippy::cast_possible_truncation)]
fn gen_iocsr_rd(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
    a: &insn_decode::ArgsR2,
    width: u64,
) -> bool {
    use gen_common::{gpr_get, gpr_set};
    use machina_accel::ir::Cond;
    let env_tmp = ctx.env;
    let pc_val = ir.new_const(Type::I64, ctx.base.pc_next - 4);
    ir.gen_mov(Type::I64, ctx.pc, pc_val);
    gen_guest_sensitive_check(ctx, ir);
    let chk = ir.new_temp(Type::I64);
    ir.gen_call(
        chk,
        helpers::loongarch_helper_check_plv as *const () as u64,
        &[env_tmp],
    );
    let zero = ir.new_const(Type::I64, 0);
    let label_ok = ir.new_label();
    ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
    ir.gen_mov(Type::I64, ctx.pc, chk);
    ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
    ir.gen_set_label(label_ok);
    let addr = gpr_get(&ctx.gpr, ir, a.rj as u8);
    let w = ir.new_const(Type::I64, width);
    let d = ir.new_temp(Type::I64);
    ir.gen_call(
        d,
        helpers::loongarch_helper_iocsrrd as *const () as u64,
        &[env_tmp, addr, w],
    );
    gpr_set(&ctx.gpr, ir, a.rd as u8, d);
    true
}

#[allow(clippy::cast_possible_truncation)]
fn gen_iocsr_wr(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
    a: &insn_decode::ArgsR2,
    width: u64,
) -> bool {
    use gen_common::gpr_get;
    use machina_accel::ir::Cond;
    let env_tmp = ctx.env;
    let pc_val = ir.new_const(Type::I64, ctx.base.pc_next - 4);
    ir.gen_mov(Type::I64, ctx.pc, pc_val);
    gen_guest_sensitive_check(ctx, ir);
    let chk = ir.new_temp(Type::I64);
    ir.gen_call(
        chk,
        helpers::loongarch_helper_check_plv as *const () as u64,
        &[env_tmp],
    );
    let zero = ir.new_const(Type::I64, 0);
    let label_ok = ir.new_label();
    ir.gen_brcond(Type::I64, chk, zero, Cond::Eq, label_ok);
    ir.gen_mov(Type::I64, ctx.pc, chk);
    ir.gen_exit_tb(EXCP_LOONGARCH_DONE);
    ir.gen_set_label(label_ok);
    let addr = gpr_get(&ctx.gpr, ir, a.rj as u8);
    let val = gpr_get(&ctx.gpr, ir, a.rd as u8);
    let w = ir.new_const(Type::I64, width);
    let d = ir.new_temp(Type::I64);
    ir.gen_call(
        d,
        helpers::loongarch_helper_iocsrwr as *const () as u64,
        &[env_tmp, addr, val, w],
    );
    true
}

fn decode_insn(
    ctx: &mut LoongArchDisasContext,
    ir: &mut Context,
    insn: u32,
) -> bool {
    insn_decode::decode(ctx, ir, insn)
}

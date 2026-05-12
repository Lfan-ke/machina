#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::missing_const_for_fn
)]

use super::super::cpu::LoongArchCpu;
use super::super::csr::{
    valid_gcsr, CRMD_DA, CRMD_PG, CRMD_PLV_MASK, CSR_CRMD, CSR_ERA, CSR_PRMD,
    CSR_TLBRERA, CSR_TLBRPRMD, GSTAT_PVM,
};
use super::super::exception::{
    ECODE_BCE, ECODE_FPD, ECODE_GSPR, ECODE_INE, ECODE_IPE,
};
use super::super::mmu::AccessType;
use machina_softfloat::{ExcFlags, Float32, Float64, FloatEnv, RoundMode};
use std::sync::atomic::{AtomicU64, Ordering};

static RDTIME_COUNTER: AtomicU64 = AtomicU64::new(0);
const RDTIME_STEP: u64 = 1_000_000;

fn enter_exception(
    cpu: &mut LoongArchCpu,
    ecode: u64,
    esubcode: u64,
    badv: Option<u64>,
) -> u64 {
    cpu.enter_exception(ecode, esubcode, badv)
}

fn clear_ll_sc_reservation(cpu: &mut LoongArchCpu) {
    cpu.llbctl = 0;
    cpu.ll_res_addr = u64::MAX;
    cpu.ll_res_val = 0;
}

fn translate_store_for_helper(
    cpu: &mut LoongArchCpu,
    addr: u64,
) -> Result<u64, ()> {
    let fault_pc = cpu.fault_pc_val();
    match cpu.translate_address_or_exception(addr, AccessType::Store, fault_pc)
    {
        Ok(pa) => Ok(pa),
        Err(vector) => {
            cpu.set_pc(vector);
            Err(())
        }
    }
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_rdtime_d(_env: *mut u8) -> u64 {
    RDTIME_COUNTER.fetch_add(RDTIME_STEP, Ordering::Relaxed) + RDTIME_STEP
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_tid(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    cpu.csr_read(super::super::csr::CSR_TID)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_div_d(a: i64, b: i64) -> i64 {
    let divisor = if b == 0 || (a == i64::MIN && b == -1) {
        1
    } else {
        b
    };
    a / divisor
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_mod_d(a: i64, b: i64) -> i64 {
    let divisor = if b == 0 || (a == i64::MIN && b == -1) {
        1
    } else {
        b
    };
    a % divisor
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_div_du(a: u64, b: u64) -> u64 {
    let divisor = if b == 0 { 1 } else { b };
    a / divisor
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_mod_du(a: u64, b: u64) -> u64 {
    let divisor = if b == 0 { 1 } else { b };
    a % divisor
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_div_w(a: i64, b: i64) -> i64 {
    let a32 = i64::from(a as i32);
    let b32 = i64::from(b as i32);
    let divisor = if b32 == 0 { 1 } else { b32 };
    i64::from((a32 / divisor) as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_mod_w(a: i64, b: i64) -> i64 {
    let a32 = i64::from(a as i32);
    let b32 = i64::from(b as i32);
    let divisor = if b32 == 0 { 1 } else { b32 };
    i64::from((a32 % divisor) as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_div_wu(a: u64, b: u64) -> i64 {
    let a32 = u64::from(a as u32);
    let b32 = u64::from(b as u32);
    let divisor = if b32 == 0 { 1 } else { b32 };
    let result = (a32 / divisor) as u32;
    i64::from(result as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_mod_wu(a: u64, b: u64) -> i64 {
    let a32 = u64::from(a as u32);
    let b32 = u64::from(b as u32);
    let divisor = if b32 == 0 { 1 } else { b32 };
    let result = (a32 % divisor) as u32;
    i64::from(result as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_mulh_d(a: i64, b: i64) -> i64 {
    ((i128::from(a) * i128::from(b)) >> 64) as i64
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_mulh_du(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) >> 64) as u64
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_mulh_w(a: i64, b: i64) -> i64 {
    let a32 = a as i32;
    let b32 = b as i32;
    let product = i64::from(a32) * i64::from(b32);
    i64::from((product >> 32) as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_mulh_wu(a: u64, b: u64) -> i64 {
    let a32 = a as u32;
    let b32 = b as u32;
    let product = u64::from(a32) * u64::from(b32);
    i64::from((product >> 32) as u32 as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_revb_2h(a: u64) -> i64 {
    let lo = a as u32;
    let swapped = ((lo & 0x00FF_00FF) << 8) | ((lo & 0xFF00_FF00) >> 8);
    i64::from(swapped as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_revb_4h(a: u64) -> u64 {
    ((a & 0x00FF_00FF_00FF_00FF) << 8) | ((a & 0xFF00_FF00_FF00_FF00) >> 8)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_revb_2w(a: u64) -> u64 {
    let lo = (a as u32).swap_bytes();
    let hi = ((a >> 32) as u32).swap_bytes();
    u64::from(hi) << 32 | u64::from(lo)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_revh_2w(a: u64) -> i64 {
    let lo = a as u32;
    let swapped = lo.rotate_right(16);
    i64::from(swapped as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_revh_d(a: u64) -> u64 {
    ((a & 0x0000_0000_0000_FFFF) << 48)
        | ((a & 0x0000_0000_FFFF_0000) << 16)
        | ((a & 0x0000_FFFF_0000_0000) >> 16)
        | ((a & 0xFFFF_0000_0000_0000) >> 48)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_bitrev_4b(a: u64) -> i64 {
    let mut v = a as u32;
    v = ((v & 0x5555_5555) << 1) | ((v & 0xAAAA_AAAA) >> 1);
    v = ((v & 0x3333_3333) << 2) | ((v & 0xCCCC_CCCC) >> 2);
    v = ((v & 0x0F0F_0F0F) << 4) | ((v & 0xF0F0_F0F0) >> 4);
    i64::from(v as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_bitrev_8b(a: u64) -> u64 {
    let mut v = a;
    v = ((v & 0x5555_5555_5555_5555) << 1) | ((v & 0xAAAA_AAAA_AAAA_AAAA) >> 1);
    v = ((v & 0x3333_3333_3333_3333) << 2) | ((v & 0xCCCC_CCCC_CCCC_CCCC) >> 2);
    v = ((v & 0x0F0F_0F0F_0F0F_0F0F) << 4) | ((v & 0xF0F0_F0F0_F0F0_F0F0) >> 4);
    v
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_bitrev_w(a: u64) -> i64 {
    let v = (a as u32).reverse_bits();
    i64::from(v as i32)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_bitrev_d(a: u64) -> u64 {
    a.reverse_bits()
}

/// Returns 0 if EUEN.FPE=1 (FP enabled). Otherwise raises FPD, stores the
/// exception vector in `cpu.pc`, and returns 1.
///
/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_check_fpe(env: *mut u8) -> u64 {
    use super::super::csr::EUEN_FPE;
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    if cpu.euen & EUEN_FPE != 0 {
        return 0;
    }
    let vector = enter_exception(cpu, u64::from(ECODE_FPD), 0, None);
    cpu.set_pc(vector);
    1
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_movfcsr2gr(env: *mut u8) -> u64 {
    let cpu = &*(env.cast::<super::super::cpu::LoongArchCpu>());
    u64::from(cpu.read_fcsr())
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_movfcsr2gr_idx(
    env: *mut u8,
    fcsrs: u64,
) -> u64 {
    let cpu = &*(env.cast::<super::super::cpu::LoongArchCpu>());
    u64::from(cpu.read_fcsr_subregister(fcsrs as u32))
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_movgr2fcsr(
    env: *mut u8,
    val: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.write_fcsr(val as u32);
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_movgr2fcsr_idx(
    env: *mut u8,
    val: u64,
    fcsrd: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.write_fcsr_subregister(fcsrd as u32, val as u32);
    0
}

const fn nanbox_s(bits: u32) -> u64 {
    0xffff_ffff_0000_0000 | bits as u64
}

fn negate_fused_result_s(value: f32) -> u64 {
    let bits = if value.is_nan() {
        value.to_bits()
    } else {
        (-value).to_bits()
    };
    nanbox_s(bits)
}

fn negate_fused_result_d(value: f64) -> u64 {
    if value.is_nan() {
        value.to_bits()
    } else {
        (-value).to_bits()
    }
}

fn fused_result_s(
    fa: f32,
    fb: f32,
    fc: f32,
    negate_c: bool,
    negate_result: bool,
) -> u64 {
    let addend = if negate_c && !fc.is_nan() { -fc } else { fc };
    let result = fa.mul_add(fb, addend);
    if negate_result {
        negate_fused_result_s(result)
    } else {
        nanbox_s(result.to_bits())
    }
}

fn fused_result_d(
    fa: f64,
    fb: f64,
    fc: f64,
    negate_c: bool,
    negate_result: bool,
) -> u64 {
    let addend = if negate_c && !fc.is_nan() { -fc } else { fc };
    let result = fa.mul_add(fb, addend);
    if negate_result {
        negate_fused_result_d(result)
    } else {
        result.to_bits()
    }
}

fn soft_negated_c_s(bits: u32, negate_c: bool) -> Float32 {
    if negate_c && !f32::from_bits(bits).is_nan() {
        Float32::from_bits(bits ^ 0x8000_0000)
    } else {
        Float32::from_bits(bits)
    }
}

fn soft_negated_c_d(bits: u64, negate_c: bool) -> Float64 {
    if negate_c && !f64::from_bits(bits).is_nan() {
        Float64::from_bits(bits ^ 0x8000_0000_0000_0000)
    } else {
        Float64::from_bits(bits)
    }
}

fn soft_negated_result_s(result: Float32, negate_result: bool) -> Float32 {
    if negate_result && !result.is_nan() {
        Float32::from_bits(result.to_bits() ^ 0x8000_0000)
    } else {
        result
    }
}

fn soft_negated_result_d(result: Float64, negate_result: bool) -> Float64 {
    if negate_result && !result.is_nan() {
        Float64::from_bits(result.to_bits() ^ 0x8000_0000_0000_0000)
    } else {
        result
    }
}

fn soft_fused_s(
    cpu: &LoongArchCpu,
    a: u64,
    b: u64,
    c: u64,
    negate_c: bool,
    negate_result: bool,
) -> (Float32, FloatEnv) {
    let mut env = soft_env(cpu);
    let result = machina_softfloat::ops::fma::fma(
        Float32::from_bits(a as u32),
        Float32::from_bits(b as u32),
        soft_negated_c_s(c as u32, negate_c),
        &mut env,
    );
    (soft_negated_result_s(result, negate_result), env)
}

fn soft_fused_d(
    cpu: &LoongArchCpu,
    a: u64,
    b: u64,
    c: u64,
    negate_c: bool,
    negate_result: bool,
) -> (Float64, FloatEnv) {
    let mut env = soft_env(cpu);
    let result = machina_softfloat::ops::fma::fma(
        Float64::from_bits(a),
        Float64::from_bits(b),
        soft_negated_c_d(c, negate_c),
        &mut env,
    );
    (soft_negated_result_d(result, negate_result), env)
}

const FP_I: u32 = 1;
const FP_U: u32 = 2;
const FP_O: u32 = 4;
const FP_Z: u32 = 8;
const FP_V: u32 = 16;

fn loongarch_round_mode(cpu: &LoongArchCpu) -> RoundMode {
    match cpu.fcsr_rounding_mode() {
        1 => RoundMode::ToZero,
        2 => RoundMode::Up,
        3 => RoundMode::Down,
        _ => RoundMode::NearEven,
    }
}

fn fpint_round_to_soft(mode: FpIntRound) -> RoundMode {
    match mode {
        FpIntRound::Down => RoundMode::Down,
        FpIntRound::Up => RoundMode::Up,
        FpIntRound::Zero => RoundMode::ToZero,
        FpIntRound::NearestEven => RoundMode::NearEven,
    }
}

fn soft_env(cpu: &LoongArchCpu) -> FloatEnv {
    FloatEnv::new(loongarch_round_mode(cpu))
}

fn soft_env_for(mode: FpIntRound) -> FloatEnv {
    FloatEnv::new(fpint_round_to_soft(mode))
}

fn soft_flags(env: &FloatEnv) -> u32 {
    let flags = env.flags();
    let mut result = 0;
    if flags.contains(ExcFlags::INEXACT) {
        result |= FP_I;
    }
    if flags.contains(ExcFlags::UNDERFLOW) {
        result |= FP_U;
    }
    if flags.contains(ExcFlags::OVERFLOW) {
        result |= FP_O;
    }
    if flags.contains(ExcFlags::DIVBYZERO) {
        result |= FP_Z;
    }
    if flags.contains(ExcFlags::INVALID) {
        result |= FP_V;
    }
    result
}

fn finish_soft_s(
    cpu: &mut LoongArchCpu,
    fd: u64,
    result: Float32,
    env: &FloatEnv,
    pc: u64,
) -> u64 {
    finish_fpr(cpu, fd, nanbox_s(result.to_bits()), soft_flags(env), pc)
}

fn finish_soft_d(
    cpu: &mut LoongArchCpu,
    fd: u64,
    result: Float64,
    env: &FloatEnv,
    pc: u64,
) -> u64 {
    finish_fpr(cpu, fd, result.to_bits(), soft_flags(env), pc)
}

fn finish_fpr(
    cpu: &mut LoongArchCpu,
    fd: u64,
    result: u64,
    flags: u32,
    pc: u64,
) -> u64 {
    if let Some(vector) = cpu.update_fcsr_exception(flags, pc) {
        cpu.set_pc(vector);
        return 1;
    }
    cpu.write_fpr(fd as usize, result);
    0
}

fn finish_fcc(
    cpu: &mut LoongArchCpu,
    cd: u64,
    result: u64,
    flags: u32,
    pc: u64,
) -> u64 {
    if let Some(vector) = cpu.update_fcsr_exception(flags, pc) {
        cpu.set_pc(vector);
        return 1;
    }
    cpu.write_fcc(cd as usize, result as u8);
    0
}

fn f32_bits_are_nan(bits: u32) -> bool {
    (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0
}

fn f64_bits_are_nan(bits: u64) -> bool {
    (bits & 0x7ff0_0000_0000_0000) == 0x7ff0_0000_0000_0000
        && (bits & 0x000f_ffff_ffff_ffff) != 0
}

fn f32_bits_are_snan(bits: u32) -> bool {
    f32_bits_are_nan(bits) && (bits & 0x0040_0000) == 0
}

fn f64_bits_are_snan(bits: u64) -> bool {
    f64_bits_are_nan(bits) && (bits & 0x0008_0000_0000_0000) == 0
}

fn quiet_nan_s(bits: u32) -> u32 {
    if f32_bits_are_nan(bits) {
        bits | 0x0040_0000
    } else {
        bits
    }
}

fn quiet_nan_d(bits: u64) -> u64 {
    if f64_bits_are_nan(bits) {
        bits | 0x0008_0000_0000_0000
    } else {
        bits
    }
}

fn flags_cmp_s(a: f32, b: f32) -> u32 {
    if f32_bits_are_snan(a.to_bits()) || f32_bits_are_snan(b.to_bits()) {
        FP_V
    } else {
        0
    }
}

fn flags_cmp_d(a: f64, b: f64) -> u32 {
    if f64_bits_are_snan(a.to_bits()) || f64_bits_are_snan(b.to_bits()) {
        FP_V
    } else {
        0
    }
}

fn flags_cmp_signal_s(a: f32, b: f32) -> u32 {
    if f32_bits_are_nan(a.to_bits()) || f32_bits_are_nan(b.to_bits()) {
        FP_V
    } else {
        0
    }
}

fn flags_cmp_signal_d(a: f64, b: f64) -> u32 {
    if f64_bits_are_nan(a.to_bits()) || f64_bits_are_nan(b.to_bits()) {
        FP_V
    } else {
        0
    }
}

fn fp_to_i32_bits_flags(value: f64, mode: FpIntRound) -> (u64, u32) {
    if value.is_nan()
        || value > f64::from(i32::MAX)
        || value < f64::from(i32::MIN)
    {
        return (0, FP_V);
    }
    let rounded = round_fp_to_int(value, mode);
    let flags = if rounded != value { FP_I } else { 0 };
    ((rounded as i32 as u32) as u64, flags)
}

fn fp_to_i64_bits_flags(value: f64, mode: FpIntRound) -> (u64, u32) {
    if value.is_nan() || value > i64::MAX as f64 || value < i64::MIN as f64 {
        return (0, FP_V);
    }
    let rounded = round_fp_to_int(value, mode);
    let flags = if rounded != value { FP_I } else { 0 };
    ((rounded as i64) as u64, flags)
}

fn dynamic_round(cpu: &LoongArchCpu) -> FpIntRound {
    match cpu.fcsr_rounding_mode() {
        1 => FpIntRound::Zero,
        2 => FpIntRound::Up,
        3 => FpIntRound::Down,
        _ => FpIntRound::NearestEven,
    }
}

fn round_f64_to_f32_bits(value: f64, mode: FpIntRound) -> (u64, u32) {
    let mut env = soft_env_for(mode);
    let result = machina_softfloat::ops::convert::convert::<Float64, Float32>(
        Float64::from_bits(value.to_bits()),
        &mut env,
    );
    (nanbox_s(result.to_bits()), soft_flags(&env))
}

fn ffint_s_w_bits(value: i32, mode: FpIntRound) -> (u64, u32) {
    let mut env = soft_env_for(mode);
    let result =
        machina_softfloat::ops::convert::from_i32::<Float32>(value, &mut env);
    (nanbox_s(result.to_bits()), soft_flags(&env))
}

fn ffint_s_l_bits(value: i64, mode: FpIntRound) -> (u64, u32) {
    let mut env = soft_env_for(mode);
    let result =
        machina_softfloat::ops::convert::from_i64::<Float32>(value, &mut env);
    (nanbox_s(result.to_bits()), soft_flags(&env))
}

fn ffint_d_w_bits(value: i32, mode: FpIntRound) -> (u64, u32) {
    let mut env = soft_env_for(mode);
    let result =
        machina_softfloat::ops::convert::from_i32::<Float64>(value, &mut env);
    (result.to_bits(), soft_flags(&env))
}

fn ffint_d_l_bits(value: i64, mode: FpIntRound) -> (u64, u32) {
    let mut env = soft_env_for(mode);
    let result =
        machina_softfloat::ops::convert::from_i64::<Float64>(value, &mut env);
    (result.to_bits(), soft_flags(&env))
}

fn fclass_s_bits(bits: u32) -> u64 {
    let sign = bits & 0x8000_0000 != 0;
    let exp = bits & 0x7f80_0000;
    let frac = bits & 0x007f_ffff;

    if exp == 0x7f80_0000 {
        if frac == 0 {
            if sign {
                1 << 2
            } else {
                1 << 6
            }
        } else if bits & 0x0040_0000 != 0 {
            1 << 1
        } else {
            1 << 0
        }
    } else if exp == 0 {
        if frac == 0 {
            if sign {
                1 << 5
            } else {
                1 << 9
            }
        } else if sign {
            1 << 4
        } else {
            1 << 8
        }
    } else if sign {
        1 << 3
    } else {
        1 << 7
    }
}

fn fclass_d_bits(bits: u64) -> u64 {
    let sign = bits & 0x8000_0000_0000_0000 != 0;
    let exp = bits & 0x7ff0_0000_0000_0000;
    let frac = bits & 0x000f_ffff_ffff_ffff;

    if exp == 0x7ff0_0000_0000_0000 {
        if frac == 0 {
            if sign {
                1 << 2
            } else {
                1 << 6
            }
        } else if bits & 0x0008_0000_0000_0000 != 0 {
            1 << 1
        } else {
            1 << 0
        }
    } else if exp == 0 {
        if frac == 0 {
            if sign {
                1 << 5
            } else {
                1 << 9
            }
        } else if sign {
            1 << 4
        } else {
            1 << 8
        }
    } else if sign {
        1 << 3
    } else {
        1 << 7
    }
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_asrtgt_d(
    env: *mut u8,
    rj: u64,
    rk: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    if rj <= rk {
        cpu.set_pc(pc);
        let vector = enter_exception(cpu, u64::from(ECODE_BCE), 0, Some(rj));
        cpu.set_pc(vector);
        return 1;
    }
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_asrtle_d(
    env: *mut u8,
    rj: u64,
    rk: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    if rj > rk {
        cpu.set_pc(pc);
        let vector = enter_exception(cpu, u64::from(ECODE_BCE), 0, Some(rj));
        cpu.set_pc(vector);
        return 1;
    }
    0
}

fn max_min_num_s_bits(
    a: u32,
    b: u32,
    is_max: bool,
    use_abs: bool,
) -> (u32, u32) {
    let flags = if f32_bits_are_snan(a) || f32_bits_are_snan(b) {
        FP_V
    } else {
        0
    };
    let a_nan = f32_bits_are_nan(a);
    let b_nan = f32_bits_are_nan(b);
    if a_nan && b_nan {
        return (quiet_nan_s(a), flags);
    }
    if a_nan {
        return (b, flags);
    }
    if b_nan {
        return (a, flags);
    }

    let fa = f32::from_bits(a);
    let fb = f32::from_bits(b);
    let ka = if use_abs { fa.abs() } else { fa };
    let kb = if use_abs { fb.abs() } else { fb };
    let choose_a = if ka == kb {
        if use_abs {
            if is_max {
                fa >= fb
            } else {
                fa <= fb
            }
        } else if fa == 0.0 && fb == 0.0 {
            if is_max {
                !fa.is_sign_negative()
            } else {
                fa.is_sign_negative()
            }
        } else {
            true
        }
    } else if is_max {
        ka > kb
    } else {
        ka < kb
    };

    (if choose_a { a } else { b }, flags)
}

fn max_min_num_d_bits(
    a: u64,
    b: u64,
    is_max: bool,
    use_abs: bool,
) -> (u64, u32) {
    let flags = if f64_bits_are_snan(a) || f64_bits_are_snan(b) {
        FP_V
    } else {
        0
    };
    let a_nan = f64_bits_are_nan(a);
    let b_nan = f64_bits_are_nan(b);
    if a_nan && b_nan {
        return (quiet_nan_d(a), flags);
    }
    if a_nan {
        return (b, flags);
    }
    if b_nan {
        return (a, flags);
    }

    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    let ka = if use_abs { fa.abs() } else { fa };
    let kb = if use_abs { fb.abs() } else { fb };
    let choose_a = if ka == kb {
        if use_abs {
            if is_max {
                fa >= fb
            } else {
                fa <= fb
            }
        } else if fa == 0.0 && fb == 0.0 {
            if is_max {
                !fa.is_sign_negative()
            } else {
                fa.is_sign_negative()
            }
        } else {
            true
        }
    } else if is_max {
        ka > kb
    } else {
        ka < kb
    };

    (if choose_a { a } else { b }, flags)
}

fn finish_unary_conversion(
    env: *mut u8,
    fd: u64,
    result: u64,
    flags: u32,
    pc: u64,
) -> u64 {
    let cpu = unsafe { &mut *(env.cast::<LoongArchCpu>()) };
    finish_fpr(cpu, fd, result, flags, pc)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fadd_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    nanbox_s((fa + fb).to_bits())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fadd_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    (fa + fb).to_bits()
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fsub_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    nanbox_s((fa - fb).to_bits())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fsub_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    (fa - fb).to_bits()
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fmul_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    nanbox_s((fa * fb).to_bits())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fmul_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    (fa * fb).to_bits()
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fdiv_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    nanbox_s((fa / fb).to_bits())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fdiv_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    (fa / fb).to_bits()
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fsqrt_s(a: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    nanbox_s(fa.sqrt().to_bits())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fsqrt_d(a: u64) -> u64 {
    let fa = f64::from_bits(a);
    fa.sqrt().to_bits()
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_caf_s(_a: u64, _b: u64) -> u64 {
    0
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_caf_d(_a: u64, _b: u64) -> u64 {
    0
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_ceq_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(fa == fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_ceq_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(fa == fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_clt_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(fa < fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_clt_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(fa < fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cle_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(fa <= fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cle_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(fa <= fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cun_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(fa.is_nan() || fb.is_nan())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cun_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(fa.is_nan() || fb.is_nan())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cueq_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(fa.is_nan() || fb.is_nan() || fa == fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cueq_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(fa.is_nan() || fb.is_nan() || fa == fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cult_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(fa.is_nan() || fb.is_nan() || fa < fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cult_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(fa.is_nan() || fb.is_nan() || fa < fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cule_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(fa.is_nan() || fb.is_nan() || fa <= fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cule_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(fa.is_nan() || fb.is_nan() || fa <= fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cne_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(!fa.is_nan() && !fb.is_nan() && fa != fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cne_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(!fa.is_nan() && !fb.is_nan() && fa != fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cor_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(!fa.is_nan() && !fb.is_nan())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cor_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(!fa.is_nan() && !fb.is_nan())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cune_s(a: u64, b: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    u64::from(fa.is_nan() || fb.is_nan() || fa != fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcmp_cune_d(a: u64, b: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    u64::from(fa.is_nan() || fb.is_nan() || fa != fb)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fmadd_s(a: u64, b: u64, c: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let fc = f32::from_bits(c as u32);
    fused_result_s(fa, fb, fc, false, false)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fmadd_d(a: u64, b: u64, c: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    let fc = f64::from_bits(c);
    fused_result_d(fa, fb, fc, false, false)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fmsub_s(a: u64, b: u64, c: u64) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let fc = f32::from_bits(c as u32);
    fused_result_s(fa, fb, fc, true, false)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fmsub_d(a: u64, b: u64, c: u64) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    let fc = f64::from_bits(c);
    fused_result_d(fa, fb, fc, true, false)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fnmadd_s(
    a: u64,
    b: u64,
    c: u64,
) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let fc = f32::from_bits(c as u32);
    fused_result_s(fa, fb, fc, false, true)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fnmadd_d(
    a: u64,
    b: u64,
    c: u64,
) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    let fc = f64::from_bits(c);
    fused_result_d(fa, fb, fc, false, true)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fnmsub_s(
    a: u64,
    b: u64,
    c: u64,
) -> u64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let fc = f32::from_bits(c as u32);
    fused_result_s(fa, fb, fc, true, true)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fnmsub_d(
    a: u64,
    b: u64,
    c: u64,
) -> u64 {
    let fa = f64::from_bits(a);
    let fb = f64::from_bits(b);
    let fc = f64::from_bits(c);
    fused_result_d(fa, fb, fc, true, true)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ffint_s_w(a: u64) -> u64 {
    let i = a as i32;
    nanbox_s((i as f32).to_bits())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ffint_d_w(a: u64) -> u64 {
    let i = a as i32;
    (f64::from(i)).to_bits()
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ffint_s_l(a: u64) -> u64 {
    let i = a as i64;
    nanbox_s((i as f32).to_bits())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ffint_d_l(a: u64) -> u64 {
    let i = a as i64;
    (i as f64).to_bits()
}

#[derive(Clone, Copy)]
enum FpIntRound {
    Down,
    Up,
    Zero,
    NearestEven,
}

fn round_nearest_even(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }

    let trunc = value.trunc();
    let frac = (value - trunc).abs();
    if frac < 0.5 {
        trunc
    } else if frac > 0.5 {
        trunc + value.signum()
    } else if (trunc as i128) % 2 == 0 {
        trunc
    } else {
        trunc + value.signum()
    }
}

fn round_fp_to_int(value: f64, mode: FpIntRound) -> f64 {
    match mode {
        FpIntRound::Down => value.floor(),
        FpIntRound::Up => value.ceil(),
        FpIntRound::Zero => value.trunc(),
        FpIntRound::NearestEven => round_nearest_even(value),
    }
}

fn fp_to_i32_bits(value: f64, mode: FpIntRound) -> u64 {
    if value.is_nan() {
        0
    } else {
        (round_fp_to_int(value, mode) as i32 as u32) as u64
    }
}

fn fp_to_i64_bits(value: f64, mode: FpIntRound) -> u64 {
    if value.is_nan() {
        0
    } else {
        (round_fp_to_int(value, mode) as i64) as u64
    }
}

fn f32_to_i32_bits(a: u64, mode: FpIntRound) -> u64 {
    fp_to_i32_bits(f64::from(f32::from_bits(a as u32)), mode)
}

fn f32_to_i64_bits(a: u64, mode: FpIntRound) -> u64 {
    fp_to_i64_bits(f64::from(f32::from_bits(a as u32)), mode)
}

fn f64_to_i32_bits(a: u64, mode: FpIntRound) -> u64 {
    fp_to_i32_bits(f64::from_bits(a), mode)
}

fn f64_to_i64_bits(a: u64, mode: FpIntRound) -> u64 {
    fp_to_i64_bits(f64::from_bits(a), mode)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrm_w_s(a: u64) -> u64 {
    f32_to_i32_bits(a, FpIntRound::Down)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrm_w_d(a: u64) -> u64 {
    f64_to_i32_bits(a, FpIntRound::Down)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrm_l_s(a: u64) -> u64 {
    f32_to_i64_bits(a, FpIntRound::Down)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrm_l_d(a: u64) -> u64 {
    f64_to_i64_bits(a, FpIntRound::Down)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrp_w_s(a: u64) -> u64 {
    f32_to_i32_bits(a, FpIntRound::Up)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrp_w_d(a: u64) -> u64 {
    f64_to_i32_bits(a, FpIntRound::Up)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrp_l_s(a: u64) -> u64 {
    f32_to_i64_bits(a, FpIntRound::Up)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrp_l_d(a: u64) -> u64 {
    f64_to_i64_bits(a, FpIntRound::Up)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrz_w_s(a: u64) -> u64 {
    f32_to_i32_bits(a, FpIntRound::Zero)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrz_w_d(a: u64) -> u64 {
    f64_to_i32_bits(a, FpIntRound::Zero)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrz_l_s(a: u64) -> u64 {
    f32_to_i64_bits(a, FpIntRound::Zero)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrz_l_d(a: u64) -> u64 {
    f64_to_i64_bits(a, FpIntRound::Zero)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrne_w_s(a: u64) -> u64 {
    f32_to_i32_bits(a, FpIntRound::NearestEven)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrne_w_d(a: u64) -> u64 {
    f64_to_i32_bits(a, FpIntRound::NearestEven)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrne_l_s(a: u64) -> u64 {
    f32_to_i64_bits(a, FpIntRound::NearestEven)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftintrne_l_d(a: u64) -> u64 {
    f64_to_i64_bits(a, FpIntRound::NearestEven)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftint_w_s(a: u64) -> u64 {
    f32_to_i32_bits(a, FpIntRound::NearestEven)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftint_w_d(a: u64) -> u64 {
    f64_to_i32_bits(a, FpIntRound::NearestEven)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftint_l_s(a: u64) -> u64 {
    f32_to_i64_bits(a, FpIntRound::NearestEven)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_ftint_l_d(a: u64) -> u64 {
    f64_to_i64_bits(a, FpIntRound::NearestEven)
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcvt_s_d(a: u64) -> u64 {
    let f = f64::from_bits(a);
    nanbox_s((f as f32).to_bits())
}

#[no_mangle]
pub extern "sysv64" fn loongarch_helper_fcvt_d_s(a: u64) -> u64 {
    let f = f32::from_bits(a as u32);
    (f64::from(f)).to_bits()
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fadd_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::add::add(
        Float32::from_bits(a as u32),
        Float32::from_bits(b as u32),
        &mut fp_env,
    );
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fadd_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::add::add(
        Float64::from_bits(a),
        Float64::from_bits(b),
        &mut fp_env,
    );
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fsub_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::add::sub(
        Float32::from_bits(a as u32),
        Float32::from_bits(b as u32),
        &mut fp_env,
    );
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fsub_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::add::sub(
        Float64::from_bits(a),
        Float64::from_bits(b),
        &mut fp_env,
    );
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fmul_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::mul::mul(
        Float32::from_bits(a as u32),
        Float32::from_bits(b as u32),
        &mut fp_env,
    );
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fmul_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::mul::mul(
        Float64::from_bits(a),
        Float64::from_bits(b),
        &mut fp_env,
    );
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fdiv_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::div::div(
        Float32::from_bits(a as u32),
        Float32::from_bits(b as u32),
        &mut fp_env,
    );
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fdiv_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::div::div(
        Float64::from_bits(a),
        Float64::from_bits(b),
        &mut fp_env,
    );
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

macro_rules! fmaxmin_s_helper {
    ($name:ident, $is_max:expr, $use_abs:expr) => {
        /// # Safety
        /// `env` must point to a valid `LoongArchCpu`.
        #[no_mangle]
        pub unsafe extern "sysv64" fn $name(
            env: *mut u8,
            fd: u64,
            a: u64,
            b: u64,
            pc: u64,
        ) -> u64 {
            let cpu = &mut *(env.cast::<LoongArchCpu>());
            let (result, flags) =
                max_min_num_s_bits(a as u32, b as u32, $is_max, $use_abs);
            finish_fpr(cpu, fd, nanbox_s(result), flags, pc)
        }
    };
}

macro_rules! fmaxmin_d_helper {
    ($name:ident, $is_max:expr, $use_abs:expr) => {
        /// # Safety
        /// `env` must point to a valid `LoongArchCpu`.
        #[no_mangle]
        pub unsafe extern "sysv64" fn $name(
            env: *mut u8,
            fd: u64,
            a: u64,
            b: u64,
            pc: u64,
        ) -> u64 {
            let cpu = &mut *(env.cast::<LoongArchCpu>());
            let (result, flags) = max_min_num_d_bits(a, b, $is_max, $use_abs);
            finish_fpr(cpu, fd, result, flags, pc)
        }
    };
}

fmaxmin_s_helper!(loongarch_helper_fmax_s_fcsr, true, false);
fmaxmin_d_helper!(loongarch_helper_fmax_d_fcsr, true, false);
fmaxmin_s_helper!(loongarch_helper_fmin_s_fcsr, false, false);
fmaxmin_d_helper!(loongarch_helper_fmin_d_fcsr, false, false);
fmaxmin_s_helper!(loongarch_helper_fmaxa_s_fcsr, true, true);
fmaxmin_d_helper!(loongarch_helper_fmaxa_d_fcsr, true, true);
fmaxmin_s_helper!(loongarch_helper_fmina_s_fcsr, false, true);
fmaxmin_d_helper!(loongarch_helper_fmina_d_fcsr, false, true);

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fscaleb_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let exp = (b as i32).clamp(-0x200, 0x200);
    let result = Float32::from_bits(a as u32).scalbn(exp, &mut fp_env);
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fscaleb_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let exp = (b as i64).clamp(-0x1000, 0x1000) as i32;
    let result = Float64::from_bits(a).scalbn(exp, &mut fp_env);
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fcopysign_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let result = (a as u32 & 0x7fff_ffff) | (b as u32 & 0x8000_0000);
    finish_fpr(cpu, fd, nanbox_s(result), 0, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fcopysign_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let result = (a & 0x7fff_ffff_ffff_ffff) | (b & 0x8000_0000_0000_0000);
    finish_fpr(cpu, fd, result, 0, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fsqrt_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::sqrt::sqrt(
        Float32::from_bits(a as u32),
        &mut fp_env,
    );
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fsqrt_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result =
        machina_softfloat::ops::sqrt::sqrt(Float64::from_bits(a), &mut fp_env);
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_frecip_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::div::div(
        Float32::from_bits(1.0f32.to_bits()),
        Float32::from_bits(a as u32),
        &mut fp_env,
    );
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_frecip_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::div::div(
        Float64::from_bits(1.0f64.to_bits()),
        Float64::from_bits(a),
        &mut fp_env,
    );
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_frsqrt_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let sqrt = machina_softfloat::ops::sqrt::sqrt(
        Float32::from_bits(a as u32),
        &mut fp_env,
    );
    let result = machina_softfloat::ops::div::div(
        Float32::from_bits(1.0f32.to_bits()),
        sqrt,
        &mut fp_env,
    );
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_frsqrt_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let sqrt =
        machina_softfloat::ops::sqrt::sqrt(Float64::from_bits(a), &mut fp_env);
    let result = machina_softfloat::ops::div::div(
        Float64::from_bits(1.0f64.to_bits()),
        sqrt,
        &mut fp_env,
    );
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_frint_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::round::round_to_int(
        Float32::from_bits(a as u32),
        &mut fp_env,
    );
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_frint_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::round::round_to_int(
        Float64::from_bits(a),
        &mut fp_env,
    );
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_flogb_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let bits = a as u32;
    let exp = bits & 0x7f80_0000;
    let frac = bits & 0x007f_ffff;
    let sign = bits & 0x8000_0000 != 0;
    let (result, flags) = if f32_bits_are_nan(bits) {
        (
            quiet_nan_s(bits),
            if f32_bits_are_snan(bits) { FP_V } else { 0 },
        )
    } else if exp == 0 && frac == 0 {
        (f32::NEG_INFINITY.to_bits(), FP_Z)
    } else if sign {
        (0x7fc0_0000, FP_V)
    } else if exp == 0x7f80_0000 {
        (f32::INFINITY.to_bits(), 0)
    } else {
        let exponent = if exp == 0 {
            let highest = 31 - frac.leading_zeros();
            highest as i32 - 149
        } else {
            ((exp >> 23) as i32) - 127
        };
        ((exponent as f32).to_bits(), 0)
    };
    finish_fpr(cpu, fd, nanbox_s(result), flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_flogb_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let exp = a & 0x7ff0_0000_0000_0000;
    let frac = a & 0x000f_ffff_ffff_ffff;
    let sign = a & 0x8000_0000_0000_0000 != 0;
    let (result, flags) = if f64_bits_are_nan(a) {
        (quiet_nan_d(a), if f64_bits_are_snan(a) { FP_V } else { 0 })
    } else if exp == 0 && frac == 0 {
        (f64::NEG_INFINITY.to_bits(), FP_Z)
    } else if sign {
        (0x7ff8_0000_0000_0000, FP_V)
    } else if exp == 0x7ff0_0000_0000_0000 {
        (f64::INFINITY.to_bits(), 0)
    } else {
        let exponent = if exp == 0 {
            let highest = 63 - frac.leading_zeros();
            highest as i32 - 1074
        } else {
            ((exp >> 52) as i32) - 1023
        };
        ((exponent as f64).to_bits(), 0)
    };
    finish_fpr(cpu, fd, result, flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fclass_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    finish_fpr(cpu, fd, fclass_s_bits(a as u32), 0, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fclass_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    finish_fpr(cpu, fd, fclass_d_bits(a), 0, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fmadd_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    c: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, fp_env) = soft_fused_s(cpu, a, b, c, false, false);
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fmadd_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    c: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, fp_env) = soft_fused_d(cpu, a, b, c, false, false);
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fmsub_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    c: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, fp_env) = soft_fused_s(cpu, a, b, c, true, false);
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fmsub_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    c: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, fp_env) = soft_fused_d(cpu, a, b, c, true, false);
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fnmadd_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    c: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, fp_env) = soft_fused_s(cpu, a, b, c, false, true);
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fnmadd_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    c: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, fp_env) = soft_fused_d(cpu, a, b, c, false, true);
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fnmsub_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    c: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, fp_env) = soft_fused_s(cpu, a, b, c, true, true);
    finish_soft_s(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fnmsub_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    b: u64,
    c: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, fp_env) = soft_fused_d(cpu, a, b, c, true, true);
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

macro_rules! fcmp_fcsr_helper {
    ($name:ident, $pure:ident, $ty:ty, $flags:ident) => {
        /// # Safety
        /// `env` must point to a valid `LoongArchCpu`.
        #[no_mangle]
        pub unsafe extern "sysv64" fn $name(
            env: *mut u8,
            cd: u64,
            a: u64,
            b: u64,
            pc: u64,
        ) -> u64 {
            let cpu = &mut *(env.cast::<LoongArchCpu>());
            let fa = <$ty>::from_bits(a as _);
            let fb = <$ty>::from_bits(b as _);
            finish_fcc(cpu, cd, $pure(a, b), $flags(fa, fb), pc)
        }
    };
}

fcmp_fcsr_helper!(
    loongarch_helper_fcmp_caf_s_fcsr,
    loongarch_helper_fcmp_caf_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_caf_d_fcsr,
    loongarch_helper_fcmp_caf_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_ceq_s_fcsr,
    loongarch_helper_fcmp_ceq_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_ceq_d_fcsr,
    loongarch_helper_fcmp_ceq_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_clt_s_fcsr,
    loongarch_helper_fcmp_clt_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_clt_d_fcsr,
    loongarch_helper_fcmp_clt_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cle_s_fcsr,
    loongarch_helper_fcmp_cle_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cle_d_fcsr,
    loongarch_helper_fcmp_cle_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cun_s_fcsr,
    loongarch_helper_fcmp_cun_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cun_d_fcsr,
    loongarch_helper_fcmp_cun_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cueq_s_fcsr,
    loongarch_helper_fcmp_cueq_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cueq_d_fcsr,
    loongarch_helper_fcmp_cueq_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cult_s_fcsr,
    loongarch_helper_fcmp_cult_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cult_d_fcsr,
    loongarch_helper_fcmp_cult_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cule_s_fcsr,
    loongarch_helper_fcmp_cule_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cule_d_fcsr,
    loongarch_helper_fcmp_cule_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cne_s_fcsr,
    loongarch_helper_fcmp_cne_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cne_d_fcsr,
    loongarch_helper_fcmp_cne_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cor_s_fcsr,
    loongarch_helper_fcmp_cor_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cor_d_fcsr,
    loongarch_helper_fcmp_cor_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cune_s_fcsr,
    loongarch_helper_fcmp_cune_s,
    f32,
    flags_cmp_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_cune_d_fcsr,
    loongarch_helper_fcmp_cune_d,
    f64,
    flags_cmp_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_saf_s_fcsr,
    loongarch_helper_fcmp_caf_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_saf_d_fcsr,
    loongarch_helper_fcmp_caf_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_seq_s_fcsr,
    loongarch_helper_fcmp_ceq_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_seq_d_fcsr,
    loongarch_helper_fcmp_ceq_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_slt_s_fcsr,
    loongarch_helper_fcmp_clt_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_slt_d_fcsr,
    loongarch_helper_fcmp_clt_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sle_s_fcsr,
    loongarch_helper_fcmp_cle_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sle_d_fcsr,
    loongarch_helper_fcmp_cle_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sun_s_fcsr,
    loongarch_helper_fcmp_cun_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sun_d_fcsr,
    loongarch_helper_fcmp_cun_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sueq_s_fcsr,
    loongarch_helper_fcmp_cueq_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sueq_d_fcsr,
    loongarch_helper_fcmp_cueq_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sult_s_fcsr,
    loongarch_helper_fcmp_cult_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sult_d_fcsr,
    loongarch_helper_fcmp_cult_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sule_s_fcsr,
    loongarch_helper_fcmp_cule_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sule_d_fcsr,
    loongarch_helper_fcmp_cule_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sne_s_fcsr,
    loongarch_helper_fcmp_cne_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sne_d_fcsr,
    loongarch_helper_fcmp_cne_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sor_s_fcsr,
    loongarch_helper_fcmp_cor_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sor_d_fcsr,
    loongarch_helper_fcmp_cor_d,
    f64,
    flags_cmp_signal_d
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sune_s_fcsr,
    loongarch_helper_fcmp_cune_s,
    f32,
    flags_cmp_signal_s
);
fcmp_fcsr_helper!(
    loongarch_helper_fcmp_sune_d_fcsr,
    loongarch_helper_fcmp_cune_d,
    f64,
    flags_cmp_signal_d
);

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ftint_w_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, flags) = fp_to_i32_bits_flags(
        f64::from(f32::from_bits(a as u32)),
        dynamic_round(cpu),
    );
    finish_fpr(cpu, fd, result, flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ftint_w_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, flags) =
        fp_to_i32_bits_flags(f64::from_bits(a), dynamic_round(cpu));
    finish_fpr(cpu, fd, result, flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ftint_l_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, flags) = fp_to_i64_bits_flags(
        f64::from(f32::from_bits(a as u32)),
        dynamic_round(cpu),
    );
    finish_fpr(cpu, fd, result, flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ftint_l_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, flags) =
        fp_to_i64_bits_flags(f64::from_bits(a), dynamic_round(cpu));
    finish_fpr(cpu, fd, result, flags, pc)
}

macro_rules! fixed_ftint_fcsr_helper {
    ($name:ident, $value:expr, $bits_flags:ident, $mode:expr) => {
        /// # Safety
        /// `env` must point to a valid `LoongArchCpu`.
        #[no_mangle]
        pub unsafe extern "sysv64" fn $name(
            env: *mut u8,
            fd: u64,
            a: u64,
            pc: u64,
        ) -> u64 {
            let (result, flags) = $bits_flags(($value)(a), $mode);
            finish_unary_conversion(env, fd, result, flags, pc)
        }
    };
}

fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrm_w_s_fcsr,
    |a| f64::from(f32::from_bits(a as u32)),
    fp_to_i32_bits_flags,
    FpIntRound::Down
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrm_w_d_fcsr,
    f64::from_bits,
    fp_to_i32_bits_flags,
    FpIntRound::Down
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrm_l_s_fcsr,
    |a| f64::from(f32::from_bits(a as u32)),
    fp_to_i64_bits_flags,
    FpIntRound::Down
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrm_l_d_fcsr,
    f64::from_bits,
    fp_to_i64_bits_flags,
    FpIntRound::Down
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrp_w_s_fcsr,
    |a| f64::from(f32::from_bits(a as u32)),
    fp_to_i32_bits_flags,
    FpIntRound::Up
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrp_w_d_fcsr,
    f64::from_bits,
    fp_to_i32_bits_flags,
    FpIntRound::Up
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrp_l_s_fcsr,
    |a| f64::from(f32::from_bits(a as u32)),
    fp_to_i64_bits_flags,
    FpIntRound::Up
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrp_l_d_fcsr,
    f64::from_bits,
    fp_to_i64_bits_flags,
    FpIntRound::Up
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrz_w_s_fcsr,
    |a| f64::from(f32::from_bits(a as u32)),
    fp_to_i32_bits_flags,
    FpIntRound::Zero
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrz_w_d_fcsr,
    f64::from_bits,
    fp_to_i32_bits_flags,
    FpIntRound::Zero
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrz_l_s_fcsr,
    |a| f64::from(f32::from_bits(a as u32)),
    fp_to_i64_bits_flags,
    FpIntRound::Zero
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrz_l_d_fcsr,
    f64::from_bits,
    fp_to_i64_bits_flags,
    FpIntRound::Zero
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrne_w_s_fcsr,
    |a| f64::from(f32::from_bits(a as u32)),
    fp_to_i32_bits_flags,
    FpIntRound::NearestEven
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrne_w_d_fcsr,
    f64::from_bits,
    fp_to_i32_bits_flags,
    FpIntRound::NearestEven
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrne_l_s_fcsr,
    |a| f64::from(f32::from_bits(a as u32)),
    fp_to_i64_bits_flags,
    FpIntRound::NearestEven
);
fixed_ftint_fcsr_helper!(
    loongarch_helper_ftintrne_l_d_fcsr,
    f64::from_bits,
    fp_to_i64_bits_flags,
    FpIntRound::NearestEven
);

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fcvt_s_d_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let (result, flags) =
        round_f64_to_f32_bits(f64::from_bits(a), dynamic_round(cpu));
    finish_fpr(cpu, fd, result, flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_fcvt_d_s_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let mut fp_env = soft_env(cpu);
    let result = machina_softfloat::ops::convert::convert::<Float32, Float64>(
        Float32::from_bits(a as u32),
        &mut fp_env,
    );
    finish_soft_d(cpu, fd, result, &fp_env, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ffint_s_w_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &*(env.cast::<LoongArchCpu>());
    let (result, flags) = ffint_s_w_bits(a as i32, dynamic_round(cpu));
    finish_unary_conversion(env, fd, result, flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ffint_d_w_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &*(env.cast::<LoongArchCpu>());
    let (result, flags) = ffint_d_w_bits(a as i32, dynamic_round(cpu));
    finish_unary_conversion(env, fd, result, flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ffint_s_l_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &*(env.cast::<LoongArchCpu>());
    let (result, flags) = ffint_s_l_bits(a as i64, dynamic_round(cpu));
    finish_unary_conversion(env, fd, result, flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ffint_d_l_fcsr(
    env: *mut u8,
    fd: u64,
    a: u64,
    pc: u64,
) -> u64 {
    let cpu = &*(env.cast::<LoongArchCpu>());
    let (result, flags) = ffint_d_l_bits(a as i64, dynamic_round(cpu));
    finish_unary_conversion(env, fd, result, flags, pc)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_iocsrrd(
    env: *mut u8,
    addr: u64,
    width: u64,
) -> u64 {
    let cpu = &*(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.iocsr_read(addr as u32, width as u32)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_iocsrwr(
    env: *mut u8,
    addr: u64,
    val: u64,
    width: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.iocsr_write(addr as u32, val, width as u32);
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_cpucfg(
    env: *mut u8,
    index: u64,
) -> u64 {
    let cpu = &*(env.cast::<super::super::cpu::LoongArchCpu>());
    match index as u32 {
        0x00 => 0x0014_C010,
        0x01 => 0x03F2_F2FE,
        0x02 => cpu.cfg.cpucfg2(),
        0x03 => 0,
        0x04 => 0x05F5_E100,
        0x05 => 0x0001_0001,
        0x06 => 0,
        0x10 => 0x0000_2C3D,
        0x11 => 0x0608_0003,
        0x12 => 0x0608_0003,
        0x13 => 0x0608_000F,
        0x14 => 0x060E_000F,
        _ => 0,
    }
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_tlbsrch(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.tlb_search_and_update();
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_tlbrd(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    let idx = (cpu.tlbidx & 0xFFF) as usize;
    cpu.tlb_read(idx);
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_tlbwr(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    let idx = (cpu.tlbidx & 0xFFF) as usize;
    cpu.tlb_write(idx);
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_tlbfill(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.tlb_fill();
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_tlbclr(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.tlb_clear_by_index();
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_tlbflush(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.tlb_flush_by_index();
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_lddir(
    env: *mut u8,
    base: u64,
    level: u64,
) -> u64 {
    let cpu = &*(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.lddir(base, level)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ldpte(
    env: *mut u8,
    base: u64,
    odd: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.ldpte(base, odd);
    0
}

/// Returns 0 on success, nonzero (exception vector) for invalid opcode.
///
/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_invtlb(
    env: *mut u8,
    op: u64,
    asid_val: u64,
    va: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let op = op as u32;
    let lvz_op = matches!(op, 0x9..=0xE | 0x10..=0x16);
    if op > 6 && !(cpu.cfg.has_lvz && lvz_op) {
        return enter_exception(cpu, u64::from(ECODE_INE), 0, None);
    }
    cpu.invtlb(op, asid_val as u16, va);
    0
}

/// Returns 0 unless the current mode is user PLV3.
/// PLV3 raises IPE and returns the exception vector.
///
/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_check_plv(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let crmd = if cpu.in_guest_mode() {
        cpu.gcsr_read(CSR_CRMD)
    } else {
        cpu.crmd
    };
    if crmd & CRMD_PLV_MASK != CRMD_PLV_MASK {
        return 0;
    }
    enter_exception(cpu, u64::from(ECODE_IPE), 0, None)
}

/// Returns 0 unless guest mode needs a sensitive-resource trap.
///
/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_check_guest_sensitive(
    env: *mut u8,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    if !cpu.in_guest_mode() {
        return 0;
    }
    enter_exception(cpu, u64::from(ECODE_GSPR), 0, None)
}

/// Returns 0 unless guest mode tries to access an invalid GCSR number.
///
/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_check_guest_csr(
    env: *mut u8,
    csr_num: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    if !cpu.in_guest_mode() || valid_gcsr(csr_num as u32) {
        return 0;
    }
    enter_exception(cpu, u64::from(ECODE_GSPR), 0, None)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_raise_exception(
    env: *mut u8,
    ecode: u64,
    esubcode: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    enter_exception(cpu, ecode, esubcode, None)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_raise_exception_with_badv(
    env: *mut u8,
    ecode: u64,
    esubcode: u64,
    badv: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    enter_exception(cpu, ecode, esubcode, Some(badv))
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ertn(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<LoongArchCpu>());
    let pc = if cpu.in_guest_mode() {
        let gtlbrera = cpu.gcsr_read(CSR_TLBRERA);
        if gtlbrera & 1 != 0 {
            let pplv_pie = cpu.gcsr_read(CSR_TLBRPRMD) & 0x7;
            let gcrmd = cpu.gcsr_read(CSR_CRMD);
            cpu.gcsr_write(
                CSR_CRMD,
                (gcrmd & !0x7 & !CRMD_DA & !CRMD_PG) | pplv_pie | CRMD_PG,
            );
            cpu.gcsr_write(CSR_TLBRERA, gtlbrera & !1);
            gtlbrera & !0x3
        } else {
            let gcrmd = cpu.gcsr_read(CSR_CRMD);
            let gprmd = cpu.gcsr_read(CSR_PRMD);
            cpu.gcsr_write(CSR_CRMD, (gcrmd & !0x7) | (gprmd & 0x7));
            cpu.gcsr_read(CSR_ERA)
        }
    } else if cpu.tlbrera & 1 != 0 {
        // Return from TLB refill: restore PLV/IE, clear DA, set PG
        let pplv_pie = cpu.tlbrprmd & 0x7;
        cpu.set_crmd(
            (cpu.crmd & !0x7 & !CRMD_DA & !CRMD_PG) | pplv_pie | CRMD_PG,
        );
        let pc = cpu.tlbrera & !0x3;
        cpu.tlbrera &= !1; // Clear ISTLBR
        pc
    } else {
        cpu.set_crmd((cpu.crmd & !0x7) | (cpu.prmd & 0x7));
        if cpu.gstat & GSTAT_PVM != 0 {
            cpu.enter_guest_mode();
        }
        cpu.era
    };
    clear_ll_sc_reservation(cpu);
    pc
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_idle(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.halted.store(true, std::sync::atomic::Ordering::Release);
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_ibar(env: *mut u8) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.request_tb_flush();
    0
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_csrrd(
    env: *mut u8,
    csr_num: u64,
) -> u64 {
    let cpu = &*(env.cast::<super::super::cpu::LoongArchCpu>());
    if cpu.in_guest_mode() {
        cpu.gcsr_read(csr_num as u32)
    } else {
        cpu.csr_read(csr_num as u32)
    }
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_csrwr(
    env: *mut u8,
    csr_num: u64,
    val: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    if cpu.in_guest_mode() {
        let old = cpu.gcsr_read(csr_num as u32);
        cpu.gcsr_write(csr_num as u32, val);
        old
    } else {
        let old = cpu.csr_read(csr_num as u32);
        cpu.csr_write(csr_num as u32, val);
        old
    }
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_csrxchg(
    env: *mut u8,
    csr_num: u64,
    val: u64,
    mask: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    if cpu.in_guest_mode() {
        cpu.gcsr_xchg(csr_num as u32, val, mask)
    } else {
        cpu.csr_xchg(csr_num as u32, val, mask)
    }
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_gcsrrd(
    env: *mut u8,
    csr_num: u64,
) -> u64 {
    let cpu = &*(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.gcsr_read(csr_num as u32)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_gcsrwr(
    env: *mut u8,
    csr_num: u64,
    val: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    let old = cpu.gcsr_read(csr_num as u32);
    cpu.gcsr_write(csr_num as u32, val);
    old
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu`.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_gcsrxchg(
    env: *mut u8,
    csr_num: u64,
    val: u64,
    mask: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    cpu.gcsr_xchg(csr_num as u32, val, mask)
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu` with correct guest_base/ram fields.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_sc_w(
    env: *mut u8,
    addr: u64,
    val: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    let llbit = cpu.llbctl;
    let res_addr = cpu.ll_res_addr;
    let res_val = cpu.ll_res_val;
    cpu.llbctl = 0;
    cpu.ll_res_addr = u64::MAX;
    cpu.ll_res_val = 0;
    if llbit == 0 || res_addr != addr {
        return 0;
    }
    let pa = match translate_store_for_helper(cpu, addr) {
        Ok(pa) => pa,
        Err(()) => return 2,
    };
    let end = match pa.checked_add(4) {
        Some(e) if pa >= cpu.ram_base && e <= cpu.ram_end => e,
        _ => return 0,
    };
    let _ = end;
    let host_ptr = (cpu.guest_base as *const u8).add(pa as usize);
    let current = (host_ptr as *const u32).read_unaligned();
    if i64::from(current as i32) != res_val as i64 {
        return 0;
    }
    let host_wptr = (cpu.guest_base as *mut u8).add(pa as usize);
    (host_wptr as *mut u32).write_unaligned(val as u32);
    1
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu` with correct guest_base/ram fields.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_helper_sc_d(
    env: *mut u8,
    addr: u64,
    val: u64,
) -> u64 {
    let cpu = &mut *(env.cast::<super::super::cpu::LoongArchCpu>());
    let llbit = cpu.llbctl;
    let res_addr = cpu.ll_res_addr;
    let res_val = cpu.ll_res_val;
    cpu.llbctl = 0;
    cpu.ll_res_addr = u64::MAX;
    cpu.ll_res_val = 0;
    if llbit == 0 || res_addr != addr {
        return 0;
    }
    let pa = match translate_store_for_helper(cpu, addr) {
        Ok(pa) => pa,
        Err(()) => return 2,
    };
    let end = match pa.checked_add(8) {
        Some(e) if pa >= cpu.ram_base && e <= cpu.ram_end => e,
        _ => return 0,
    };
    let _ = end;
    let host_ptr = (cpu.guest_base as *const u8).add(pa as usize);
    let current = (host_ptr as *const u64).read_unaligned();
    if current != res_val {
        return 0;
    }
    let host_wptr = (cpu.guest_base as *mut u8).add(pa as usize);
    (host_wptr as *mut u64).write_unaligned(val);
    1
}

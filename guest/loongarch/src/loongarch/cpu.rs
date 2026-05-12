use std::mem::offset_of;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};

use super::csr::{
    ASID_WRITE_MASK, CRMD_DA, CRMD_IE, CRMD_PG, CRMD_PLV_MASK, CSR_ASID,
    CSR_BADV, CSR_CRMD, CSR_DMW0, CSR_DMW1, CSR_DMW2, CSR_DMW3, CSR_ECFG,
    CSR_EENTRY, CSR_ERA, CSR_ESTAT, CSR_PRMD, CSR_PWCH, CSR_PWCL, CSR_STLBPS,
    CSR_TCFG, CSR_TLBEHI, CSR_TLBELO0, CSR_TLBELO1, CSR_TLBIDX, CSR_TLBRBADV,
    CSR_TLBREHI, CSR_TLBRELO0, CSR_TLBRELO1, CSR_TLBRENTRY, CSR_TLBRERA,
    CSR_TLBRPRMD, CSR_TVAL, ESTAT_IS_MASK, GSTAT_GID, GSTAT_GID_SHIFT,
    GSTAT_PVM, GTLBC_TGID, GTLBC_TGID_SHIFT, GTLBC_USETGID,
};
use super::exception::{
    ECODE_FPE, ECODE_GCM, ECODE_GSPR, ECODE_HVC, ECODE_PIF, ECODE_PIL,
    ECODE_PIS, ECODE_PME, ECODE_PNR, ECODE_PNX, ECODE_PPI, ECODE_TLBR,
    ESUBCODE_GCHC,
};
use super::ext::LoongArchCfg;
use super::mmu::{
    self, AccessType, LoongArchMmu, TlbLookupResult, FAST_TLB_MMIO_ADDEND,
    FAST_TLB_PAGE_MASK,
};

pub const NUM_GPRS: usize = 32;
pub const NUM_FPRS: usize = 32;
pub const NUM_FCC: usize = 8;
pub const NUM_DMW: usize = 4;
pub const NUM_SAVE: usize = 8;
pub const NUM_GCSR: usize = 0x184;
const TIMER_INTERRUPT_BIT: u64 = 1 << 11;
const ECFG_VS_SHIFT: u64 = 16;
const ECFG_VS_MASK: u64 = 0x7;
const EXCCODE_EXTERNAL_INT: u32 = 64;
const TLBENTRY_HUGE: u64 = 1 << 6;
const TLBENTRY_HGLOBAL: u64 = 1 << 12;
const TLBENTRY_LEVEL_SHIFT: u64 = 13;
const TLBENTRY_LEVEL_MASK: u64 = 0x3;
const HW_PTE_MASK_LA64: u64 = 0xE000_FFFF_FFFF_F1FF;
pub(crate) const FCSR_ENABLE_MASK: u32 = 0x0000_001F;
pub(crate) const FCSR_RM_MASK: u32 = 0x0000_0300;
pub(crate) const FCSR_FLAGS_MASK: u32 = 0x001F_0000;
pub(crate) const FCSR_CAUSE_MASK: u32 = 0x1F00_0000;
pub(crate) const FCSR_WRITE_MASK: u32 =
    FCSR_ENABLE_MASK | FCSR_RM_MASK | FCSR_FLAGS_MASK | FCSR_CAUSE_MASK;

#[repr(C)]
pub struct LoongArchCpu {
    pub(crate) gpr: [u64; NUM_GPRS],
    pub(crate) pc: u64,
    pub(crate) guest_base: u64,

    pub(crate) fpr: [u64; NUM_FPRS],
    pub(crate) fcsr0: u32,
    pub(crate) fcc: [u8; NUM_FCC],

    pub(crate) crmd: u64,
    pub(crate) prmd: u64,
    pub(crate) euen: u64,
    pub(crate) ecfg: u64,
    pub(crate) estat: u64,
    pub(crate) era: u64,
    pub(crate) badv: u64,
    pub(crate) badi: u64,
    pub(crate) eentry: u64,
    pub(crate) tlbidx: u64,
    pub(crate) tlbehi: u64,
    pub(crate) tlbelo0: u64,
    pub(crate) tlbelo1: u64,
    pub(crate) gtlbc: u64,
    pub(crate) trgp: u64,
    pub(crate) asid: u64,
    pub(crate) pgdl: u64,
    pub(crate) pgdh: u64,
    pub(crate) pgd: u64,
    pub(crate) pwcl: u64,
    pub(crate) pwch: u64,
    pub(crate) stlbps: u64,
    pub(crate) rvacfg: u64,
    pub(crate) cpuid: u64,
    pub(crate) prcfg1: u64,
    pub(crate) prcfg2: u64,
    pub(crate) prcfg3: u64,
    pub(crate) llbctl: u64,
    pub(crate) ll_res_addr: u64,
    pub(crate) ll_res_val: u64,
    pub(crate) tlbrentry: u64,
    pub(crate) tlbrbadv: u64,
    pub(crate) tlbrera: u64,
    pub(crate) tlbrsave: u64,
    pub(crate) tlbrelo0: u64,
    pub(crate) tlbrelo1: u64,
    pub(crate) tlbrehi: u64,
    pub(crate) tlbrprmd: u64,
    pub(crate) dmw: [u64; NUM_DMW],

    pub(crate) tcfg: u64,
    pub(crate) tval: u64,
    pub(crate) cntc: u64,
    pub(crate) ticlr: u64,
    pub(crate) tid: u64,

    pub(crate) gstat: u64,
    pub(crate) gcfg: u64,
    pub(crate) gintc: u64,
    pub(crate) gcntc: u64,

    pub(crate) save: [u64; NUM_SAVE],
    pub(crate) gcsr: [u64; NUM_GCSR],

    pub(crate) interrupt_request: AtomicU32,
    pub(crate) halted: AtomicBool,
    pub(crate) neg_align: i32,
    pub(crate) last_phys_pc: u64,

    pub(crate) mmu: LoongArchMmu,
    pub(crate) guest_mmu: Box<LoongArchMmu>,

    pub(crate) ipi_status: u64,
    pub(crate) ipi_enable: u64,
    pub(crate) ipi_mailbox: [u64; 4],
    pub(crate) iocsr_dispatcher: IocsrDispatcher,

    pub(crate) as_ptr: u64,
    pub(crate) ram_base: u64,
    pub(crate) ram_end: u64,
    pub(crate) code_pages_ptr: u64,
    pub(crate) code_pages_len: u64,
    pub(crate) tb_flush_pending: bool,
    pub(crate) translation_fault_pending: bool,
    pub(crate) mem_fault_cause: u64,
    pub(crate) fault_pc: u64,
    pub(crate) cfg: LoongArchCfg,
}

pub struct LoongArchCpuInterruptState {
    hwi_pending: AtomicU32,
    ipi_pending: AtomicBool,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
}

impl Default for LoongArchCpuInterruptState {
    fn default() -> Self {
        Self {
            hwi_pending: AtomicU32::new(0),
            ipi_pending: AtomicBool::new(false),
            wait_lock: Mutex::new(()),
            wait_cv: Condvar::new(),
        }
    }
}

impl LoongArchCpuInterruptState {
    pub fn set_hwi_interrupt_pending(&self, hwi: u8, pending: bool) {
        if hwi >= 8 {
            return;
        }
        let mask = 1_u32 << hwi;
        if pending {
            self.hwi_pending.fetch_or(mask, Ordering::Release);
        } else {
            self.hwi_pending.fetch_and(!mask, Ordering::Release);
        }
        self.wake_waiters();
    }

    pub fn set_ipi_interrupt_pending(&self, pending: bool) {
        self.ipi_pending.store(pending, Ordering::Release);
        self.wake_waiters();
    }

    #[must_use]
    pub fn has_pending_irq(&self) -> bool {
        self.hwi_pending.load(Ordering::Acquire) != 0
            || self.ipi_pending.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn wait_for_irq_or_stop(
        &self,
        running: &AtomicBool,
        run_gate: Option<&AtomicBool>,
    ) -> bool {
        let mut guard = self.wait_lock.lock().unwrap();
        loop {
            if self.has_pending_irq() {
                return true;
            }
            if !running.load(Ordering::Acquire) {
                return false;
            }
            if run_gate.is_some_and(|gate| !gate.load(Ordering::Acquire)) {
                return false;
            }
            guard = self.wait_cv.wait(guard).unwrap();
        }
    }

    pub fn wake_waiters(&self) {
        let guard = self.wait_lock.lock().unwrap();
        self.wait_cv.notify_all();
        drop(guard);
    }

    #[must_use]
    pub fn pending_interrupt(&self, cpu: &LoongArchCpu) -> bool {
        if cpu.crmd & CRMD_IE == 0 {
            return false;
        }
        cpu.masked_interrupt_line_for_estat(
            cpu.estat | self.pending_estat_bits(),
        )
        .is_some()
    }

    pub fn apply_to_cpu(&self, cpu: &mut LoongArchCpu) {
        let hwi = self.hwi_pending.load(Ordering::Acquire);
        for line in 0..8 {
            cpu.set_hwi_interrupt_pending(line, hwi & (1_u32 << line) != 0);
        }
        cpu.set_ipi_interrupt_pending(self.ipi_pending.load(Ordering::Acquire));
    }

    fn pending_estat_bits(&self) -> u64 {
        let mut bits = 0;
        let hwi = self.hwi_pending.load(Ordering::Acquire);
        for line in 0..8 {
            if hwi & (1_u32 << line) != 0 {
                bits |= 1_u64 << (2 + line);
            }
        }
        if self.ipi_pending.load(Ordering::Acquire) {
            bits |= 1_u64 << 12;
        }
        bits
    }
}

pub const GPR_OFFSET: usize = offset_of!(LoongArchCpu, gpr);
pub const PC_OFFSET: usize = offset_of!(LoongArchCpu, pc);
pub const GUEST_BASE_OFFSET: usize = offset_of!(LoongArchCpu, guest_base);
pub const FPR_OFFSET: usize = offset_of!(LoongArchCpu, fpr);
pub const FCSR0_OFFSET: usize = offset_of!(LoongArchCpu, fcsr0);
pub const FCC_OFFSET: usize = offset_of!(LoongArchCpu, fcc);
pub const CRMD_OFFSET: usize = offset_of!(LoongArchCpu, crmd);
pub const ESTAT_OFFSET: usize = offset_of!(LoongArchCpu, estat);
pub const ERA_OFFSET: usize = offset_of!(LoongArchCpu, era);
pub const BADV_OFFSET: usize = offset_of!(LoongArchCpu, badv);
pub const EENTRY_OFFSET: usize = offset_of!(LoongArchCpu, eentry);
pub const LLBCTL_OFFSET: usize = offset_of!(LoongArchCpu, llbctl);
pub const LL_RES_ADDR_OFFSET: usize = offset_of!(LoongArchCpu, ll_res_addr);
pub const LL_RES_VAL_OFFSET: usize = offset_of!(LoongArchCpu, ll_res_val);
pub const RAM_BASE_OFFSET: usize = offset_of!(LoongArchCpu, ram_base);
pub const RAM_END_OFFSET: usize = offset_of!(LoongArchCpu, ram_end);
pub const GUEST_BASE_CPU_OFFSET: usize = offset_of!(LoongArchCpu, guest_base);
pub const NEG_ALIGN_CPU_OFFSET: usize = offset_of!(LoongArchCpu, neg_align);
pub const FAST_TLB_PTR_OFFSET: usize =
    offset_of!(LoongArchCpu, mmu) + offset_of!(LoongArchMmu, fast_tlb);
pub const MEM_FAULT_CAUSE_OFFSET: usize =
    offset_of!(LoongArchCpu, mem_fault_cause);
pub const FAULT_PC_OFFSET: usize = offset_of!(LoongArchCpu, fault_pc);

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoongArchCpuLayoutOffsets {
    pub gpr: usize,
    pub pc: usize,
    pub guest_base: usize,
    pub fpr: usize,
    pub fcsr0: usize,
    pub fcc: usize,
    pub crmd: usize,
    pub estat: usize,
    pub era: usize,
    pub badv: usize,
    pub eentry: usize,
    pub llbctl: usize,
    pub ll_res_addr: usize,
    pub ll_res_val: usize,
    pub ram_base: usize,
    pub ram_end: usize,
    pub neg_align: usize,
}

#[must_use]
pub const fn gpr_offset(i: usize) -> usize {
    GPR_OFFSET + i * 8
}

#[must_use]
pub const fn fpr_offset(i: usize) -> usize {
    FPR_OFFSET + i * 8
}

const fn fcsr_subregister_mask(idx: u32) -> u32 {
    match idx {
        0 => FCSR_WRITE_MASK,
        1 => FCSR_ENABLE_MASK,
        2 => FCSR_FLAGS_MASK | FCSR_CAUSE_MASK,
        3 => FCSR_RM_MASK,
        _ => 0,
    }
}

fn mask_iocsr_width(val: u64, width: u32) -> u64 {
    match width {
        1 => val & 0xff,
        2 => val & 0xffff,
        4 => val & 0xffff_ffff,
        _ => val,
    }
}

pub type IocsrReadFn = unsafe extern "C" fn(
    opaque: *mut (),
    cpu_id: u32,
    addr: u32,
    width: u32,
    out: *mut u64,
) -> bool;
pub type IocsrWriteFn = unsafe extern "C" fn(
    opaque: *mut (),
    cpu_id: u32,
    addr: u32,
    width: u32,
    val: u64,
) -> bool;

#[derive(Clone, Copy)]
pub struct IocsrDispatcher {
    opaque: usize,
    read: Option<IocsrReadFn>,
    write: Option<IocsrWriteFn>,
}

impl IocsrDispatcher {
    #[must_use]
    pub const fn none() -> Self {
        Self {
            opaque: 0,
            read: None,
            write: None,
        }
    }

    #[must_use]
    pub fn new(
        opaque: *mut (),
        read: IocsrReadFn,
        write: IocsrWriteFn,
    ) -> Self {
        Self {
            opaque: opaque as usize,
            read: Some(read),
            write: Some(write),
        }
    }

    fn read(&self, cpu_id: u32, addr: u32, width: u32) -> Option<u64> {
        let read = self.read?;
        let mut out = 0;
        if unsafe {
            read(self.opaque as *mut (), cpu_id, addr, width, &mut out)
        } {
            Some(out)
        } else {
            None
        }
    }

    fn write(&self, cpu_id: u32, addr: u32, width: u32, val: u64) -> bool {
        let Some(write) = self.write else {
            return false;
        };
        unsafe { write(self.opaque as *mut (), cpu_id, addr, width, val) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TranslationFault {
    addr: u64,
    fault: TlbLookupResult,
    host_stage: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranslationOutcome {
    Hit { pa: u64, mat: u8 },
    Fault(TranslationFault),
}

impl LoongArchCpu {
    #[must_use]
    pub fn new() -> Self {
        Self::with_cfg(LoongArchCfg {
            has_fpu: true,
            has_lsx: false,
            has_lasx: false,
            has_lbt: false,
            has_lvz: true,
        })
    }

    #[doc(hidden)]
    #[must_use]
    pub fn layout_offsets_for_tests() -> LoongArchCpuLayoutOffsets {
        LoongArchCpuLayoutOffsets {
            gpr: offset_of!(LoongArchCpu, gpr),
            pc: offset_of!(LoongArchCpu, pc),
            guest_base: offset_of!(LoongArchCpu, guest_base),
            fpr: offset_of!(LoongArchCpu, fpr),
            fcsr0: offset_of!(LoongArchCpu, fcsr0),
            fcc: offset_of!(LoongArchCpu, fcc),
            crmd: offset_of!(LoongArchCpu, crmd),
            estat: offset_of!(LoongArchCpu, estat),
            era: offset_of!(LoongArchCpu, era),
            badv: offset_of!(LoongArchCpu, badv),
            eentry: offset_of!(LoongArchCpu, eentry),
            llbctl: offset_of!(LoongArchCpu, llbctl),
            ll_res_addr: offset_of!(LoongArchCpu, ll_res_addr),
            ll_res_val: offset_of!(LoongArchCpu, ll_res_val),
            ram_base: offset_of!(LoongArchCpu, ram_base),
            ram_end: offset_of!(LoongArchCpu, ram_end),
            neg_align: offset_of!(LoongArchCpu, neg_align),
        }
    }

    #[must_use]
    pub fn with_cfg(cfg: LoongArchCfg) -> Self {
        let mut gcsr = [0; NUM_GCSR];
        gcsr[super::csr::CSR_CRMD as usize] = 0x0000_0008;
        gcsr[super::csr::CSR_PRCFG1 as usize] = 0x72F8;
        gcsr[super::csr::CSR_PRCFG2 as usize] = 0x4020_5000;
        gcsr[super::csr::CSR_PRCFG3 as usize] = 0x0080_73F2;

        Self {
            gpr: [0; NUM_GPRS],
            pc: 0,
            guest_base: 0,
            fpr: [0; NUM_FPRS],
            fcsr0: 0,
            fcc: [0; NUM_FCC],
            crmd: 0x0000_0008,
            prmd: 0,
            euen: 0,
            ecfg: 0,
            estat: 0,
            era: 0,
            badv: 0,
            badi: 0,
            eentry: 0,
            tlbidx: 0,
            tlbehi: 0,
            tlbelo0: 0,
            tlbelo1: 0,
            gtlbc: 0,
            trgp: 0,
            asid: 0,
            pgdl: 0,
            pgdh: 0,
            pgd: 0,
            pwcl: 0,
            pwch: 0,
            stlbps: 0,
            rvacfg: 0,
            cpuid: 0,
            prcfg1: 0x72F8,
            prcfg2: 0x4020_5000,
            prcfg3: 0x0080_73F2,
            llbctl: 0,
            ll_res_addr: u64::MAX,
            ll_res_val: 0,
            tlbrentry: 0,
            tlbrbadv: 0,
            tlbrera: 0,
            tlbrsave: 0,
            tlbrelo0: 0,
            tlbrelo1: 0,
            tlbrehi: 0,
            tlbrprmd: 0,
            dmw: [0; NUM_DMW],
            tcfg: 0,
            tval: 0,
            cntc: 0,
            ticlr: 0,
            tid: 0,
            gstat: 0,
            gcfg: 0,
            gintc: 0,
            gcntc: 0,
            save: [0; NUM_SAVE],
            gcsr,
            interrupt_request: AtomicU32::new(0),
            halted: AtomicBool::new(false),
            neg_align: 0,
            last_phys_pc: 0,
            mmu: LoongArchMmu::new(),
            guest_mmu: Box::new(LoongArchMmu::new()),
            ipi_status: 0,
            ipi_enable: 0,
            ipi_mailbox: [0; 4],
            iocsr_dispatcher: IocsrDispatcher::none(),
            as_ptr: 0,
            ram_base: 0,
            ram_end: 0,
            code_pages_ptr: 0,
            code_pages_len: 0,
            tb_flush_pending: false,
            translation_fault_pending: false,
            mem_fault_cause: 0,
            fault_pc: 0,
            cfg,
        }
    }

    #[must_use]
    pub const fn crmd(&self) -> u64 {
        self.crmd
    }

    pub(crate) fn set_crmd(&mut self, val: u64) {
        let old = self.crmd;
        let was_pending = self.pending_interrupt();
        self.crmd = val;
        if (old ^ val) & (CRMD_PLV_MASK | CRMD_DA | CRMD_PG) != 0 {
            self.flush_fast_tlb();
        }
        self.wake_if_new_enabled_interrupt(was_pending);
    }

    pub(crate) fn set_asid_low(&mut self, val: u64) {
        let old = self.asid;
        self.asid = (self.asid & !ASID_WRITE_MASK) | (val & ASID_WRITE_MASK);
        if (old ^ self.asid) & ASID_WRITE_MASK != 0 {
            self.invalidate_tlb_translations();
        }
    }

    #[must_use]
    pub const fn prmd(&self) -> u64 {
        self.prmd
    }

    #[must_use]
    pub const fn estat(&self) -> u64 {
        self.estat
    }

    #[must_use]
    pub const fn era(&self) -> u64 {
        self.era
    }

    pub fn set_cpuid(&mut self, cpuid: u64) {
        self.cpuid = cpuid;
    }

    #[must_use]
    pub fn in_guest_mode(&self) -> bool {
        self.gstat & super::csr::GSTAT_VM != 0
    }

    fn guest_gid(&self) -> u8 {
        ((self.gstat & GSTAT_GID) >> GSTAT_GID_SHIFT) as u8
    }

    fn will_return_to_guest(&self) -> bool {
        self.cfg.has_lvz && !self.in_guest_mode() && self.gstat & GSTAT_PVM != 0
    }

    fn target_gid(&self) -> u8 {
        if self.in_guest_mode() {
            return self.guest_gid();
        }
        if self.gtlbc & GTLBC_USETGID != 0 {
            return ((self.gtlbc & GTLBC_TGID) >> GTLBC_TGID_SHIFT) as u8;
        }
        if self.will_return_to_guest() {
            return self.guest_gid();
        }
        0
    }

    pub(crate) fn enter_guest_mode(&mut self) {
        self.gstat =
            (self.gstat | super::csr::GSTAT_VM) & !super::csr::GSTAT_PVM;
        self.flush_fast_tlb();
    }

    pub(crate) fn leave_guest_mode_for_exception(&mut self) {
        if self.in_guest_mode() {
            self.gstat =
                (self.gstat | super::csr::GSTAT_PVM) & !super::csr::GSTAT_VM;
            self.flush_fast_tlb();
        }
    }

    pub fn set_iocsr_dispatcher(&mut self, dispatcher: IocsrDispatcher) {
        self.iocsr_dispatcher = dispatcher;
    }

    pub(crate) const fn set_era(&mut self, val: u64) {
        self.era = val;
    }

    #[must_use]
    pub const fn euen(&self) -> u64 {
        self.euen
    }

    pub(crate) const fn set_euen(&mut self, val: u64) {
        self.euen = val;
    }

    #[must_use]
    pub const fn eentry(&self) -> u64 {
        self.eentry
    }

    pub(crate) const fn set_eentry(&mut self, val: u64) {
        self.eentry = val;
    }

    #[must_use]
    pub const fn ecfg(&self) -> u64 {
        self.ecfg
    }

    pub(crate) const fn set_ecfg(&mut self, val: u64) {
        self.ecfg = val;
    }

    #[must_use]
    pub const fn pc(&self) -> u64 {
        self.pc
    }

    pub const fn set_pc(&mut self, val: u64) {
        self.pc = val;
    }

    #[must_use]
    pub const fn read_gpr(&self, idx: usize) -> u64 {
        self.gpr[idx]
    }

    pub fn write_gpr(&mut self, idx: usize, val: u64) {
        if idx != 0 {
            self.gpr[idx] = val;
        }
    }

    #[must_use]
    pub const fn read_fpr(&self, idx: usize) -> u64 {
        self.fpr[idx]
    }

    pub const fn write_fpr(&mut self, idx: usize, val: u64) {
        self.fpr[idx] = val;
    }

    #[must_use]
    pub const fn read_fcc(&self, idx: usize) -> u8 {
        self.fcc[idx]
    }

    pub const fn write_fcc(&mut self, idx: usize, val: u8) {
        self.fcc[idx] = val & 1;
    }

    #[must_use]
    pub const fn dmw(&self, idx: usize) -> u64 {
        self.dmw[idx]
    }

    #[must_use]
    pub fn masked_interrupt_line(&self) -> Option<u32> {
        if self.in_guest_mode() {
            self.masked_interrupt_line_for(
                self.gcsr[CSR_ESTAT as usize],
                self.gcsr[CSR_ECFG as usize],
            )
        } else {
            self.masked_interrupt_line_for_estat(self.estat)
        }
    }

    fn masked_interrupt_line_for_estat(&self, estat: u64) -> Option<u32> {
        self.masked_interrupt_line_for(estat, self.ecfg)
    }

    fn masked_interrupt_line_for(&self, estat: u64, ecfg: u64) -> Option<u32> {
        let pending = estat & ecfg & ESTAT_IS_MASK;
        if pending == 0 {
            None
        } else {
            Some(63_u32 - pending.leading_zeros())
        }
    }

    #[must_use]
    pub fn pending_interrupt_line(&self) -> Option<u32> {
        if self.in_guest_mode() {
            if self.gcsr[CSR_CRMD as usize] & CRMD_IE == 0 {
                return None;
            }
            return self.masked_interrupt_line();
        }
        if self.crmd & CRMD_IE == 0 {
            return None;
        }
        self.masked_interrupt_line()
    }

    #[must_use]
    pub fn pending_interrupt(&self) -> bool {
        self.pending_interrupt_line().is_some()
    }

    #[must_use]
    pub fn external_interrupt_vector(
        &self,
        irq: u32,
        default_vector: u64,
    ) -> u64 {
        let (ecfg, eentry) = if self.in_guest_mode() {
            (self.gcsr[CSR_ECFG as usize], self.gcsr[CSR_EENTRY as usize])
        } else {
            (self.ecfg, self.eentry)
        };
        let vs = (ecfg >> ECFG_VS_SHIFT) & ECFG_VS_MASK;
        if vs == 0 {
            default_vector
        } else {
            eentry.wrapping_add(
                u64::from(EXCCODE_EXTERNAL_INT + irq) * ((1_u64 << vs) * 4),
            )
        }
    }

    pub fn read_fcsr(&self) -> u32 {
        self.fcsr0
    }

    pub fn write_fcsr(&mut self, val: u32) {
        self.fcsr0 = val & FCSR_WRITE_MASK;
    }

    #[must_use]
    pub(crate) const fn fcsr_rounding_mode(&self) -> u32 {
        (self.fcsr0 & FCSR_RM_MASK) >> 8
    }

    #[must_use]
    pub(crate) const fn read_fcsr_subregister(&self, idx: u32) -> u32 {
        self.fcsr0 & fcsr_subregister_mask(idx)
    }

    pub(crate) fn write_fcsr_subregister(&mut self, idx: u32, val: u32) {
        let mask = fcsr_subregister_mask(idx);
        self.fcsr0 = (self.fcsr0 & !mask) | (val & mask);
    }

    pub(crate) fn update_fcsr_exception(
        &mut self,
        flags: u32,
        fault_pc: u64,
    ) -> Option<u64> {
        let flags = flags & FCSR_ENABLE_MASK;
        self.fcsr0 = (self.fcsr0 & !FCSR_CAUSE_MASK) | (flags << 24);
        if flags == 0 {
            return None;
        }
        if self.fcsr0 & flags != 0 {
            self.pc = fault_pc;
            return Some(self.enter_exception(u64::from(ECODE_FPE), 0, None));
        }
        self.fcsr0 |= flags << 16;
        None
    }

    #[must_use]
    pub fn mmu(&self) -> &LoongArchMmu {
        &self.mmu
    }

    pub fn mmu_mut(&mut self) -> &mut LoongArchMmu {
        &mut self.mmu
    }

    fn active_translation_asid(&self) -> u16 {
        if self.in_guest_mode() {
            (self.gcsr[CSR_ASID as usize] & 0x3FF) as u16
        } else {
            (self.asid & 0x3FF) as u16
        }
    }

    fn active_translation_stlbps(&self) -> u8 {
        if self.in_guest_mode() {
            self.gcsr[CSR_STLBPS as usize] as u8
        } else {
            self.stlbps as u8
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lookup_address_with(
        &self,
        va: u64,
        access: AccessType,
        crmd: u64,
        dmw: &[u64; NUM_DMW],
        mmu: &LoongArchMmu,
        asid: u16,
        gid: u8,
        stlb_ps: u8,
        plv: u8,
    ) -> TlbLookupResult {
        if let Some(pa) = mmu::direct_map_address_with(crmd, dmw, va) {
            return TlbLookupResult::Hit { pa, mat: 0 };
        }
        if crmd & super::csr::CRMD_PG == 0 {
            return TlbLookupResult::Miss;
        }
        mmu.tlb_lookup(va, asid, gid, stlb_ps, access, plv)
    }

    fn translate_host_address(
        &self,
        va: u64,
        access: AccessType,
    ) -> TlbLookupResult {
        self.lookup_address_with(
            va,
            access,
            self.crmd,
            &self.dmw,
            &self.mmu,
            (self.asid & 0x3FF) as u16,
            self.target_gid(),
            self.stlbps as u8,
            (self.crmd & CRMD_PLV_MASK) as u8,
        )
    }

    fn translate_guest_first_stage(
        &self,
        va: u64,
        access: AccessType,
    ) -> TlbLookupResult {
        let dmw = [
            self.gcsr[CSR_DMW0 as usize],
            self.gcsr[CSR_DMW1 as usize],
            self.gcsr[CSR_DMW2 as usize],
            self.gcsr[CSR_DMW3 as usize],
        ];
        self.lookup_address_with(
            va,
            access,
            self.gcsr[CSR_CRMD as usize],
            &dmw,
            &self.guest_mmu,
            (self.gcsr[CSR_ASID as usize] & 0x3FF) as u16,
            self.target_gid(),
            self.gcsr[CSR_STLBPS as usize] as u8,
            (self.gcsr[CSR_CRMD as usize] & CRMD_PLV_MASK) as u8,
        )
    }

    fn translate_guest_host_stage(
        &self,
        gpa: u64,
        access: AccessType,
    ) -> TlbLookupResult {
        self.mmu.tlb_lookup(
            gpa,
            (self.asid & 0x3FF) as u16,
            self.target_gid(),
            self.stlbps as u8,
            access,
            0,
        )
    }

    fn translate_address_detail(
        &self,
        va: u64,
        access: AccessType,
    ) -> TranslationOutcome {
        if !self.in_guest_mode() {
            return match self.translate_host_address(va, access) {
                TlbLookupResult::Hit { pa, mat } => {
                    TranslationOutcome::Hit { pa, mat }
                }
                fault => TranslationOutcome::Fault(TranslationFault {
                    addr: va,
                    fault,
                    host_stage: false,
                }),
            };
        }

        let gpa = match self.translate_guest_first_stage(va, access) {
            TlbLookupResult::Hit { pa, .. } => pa,
            fault => {
                return TranslationOutcome::Fault(TranslationFault {
                    addr: va,
                    fault,
                    host_stage: false,
                });
            }
        };
        match self.translate_guest_host_stage(gpa, access) {
            TlbLookupResult::Hit { pa, mat } => {
                TranslationOutcome::Hit { pa, mat }
            }
            fault => TranslationOutcome::Fault(TranslationFault {
                addr: gpa,
                fault,
                host_stage: true,
            }),
        }
    }

    #[must_use]
    pub fn translate_address(
        &self,
        va: u64,
        access: AccessType,
    ) -> TlbLookupResult {
        match self.translate_address_detail(va, access) {
            TranslationOutcome::Hit { pa, mat } => {
                TlbLookupResult::Hit { pa, mat }
            }
            TranslationOutcome::Fault(fault) => fault.fault,
        }
    }

    pub fn translate_address_and_cache(
        &mut self,
        va: u64,
        access: AccessType,
    ) -> TlbLookupResult {
        match self.translate_address_detail(va, access) {
            TranslationOutcome::Hit { pa, mat } => {
                let addend = self.fast_tlb_addend(va, pa);
                self.fill_fast_tlb_for_translation(va, access, pa, addend);
                TlbLookupResult::Hit { pa, mat }
            }
            TranslationOutcome::Fault(fault) => fault.fault,
        }
    }

    pub fn fill_fast_tlb_for_translation(
        &mut self,
        va: u64,
        access: AccessType,
        pa: u64,
        addend: usize,
    ) {
        self.active_tlb_mmu_mut()
            .fill_fast_tlb(va, access, pa, addend);
    }

    pub fn translate_address_or_exception(
        &mut self,
        va: u64,
        access: AccessType,
        fault_pc: u64,
    ) -> Result<u64, u64> {
        match self.translate_address_detail(va, access) {
            TranslationOutcome::Hit { pa, .. } => Ok(pa),
            TranslationOutcome::Fault(fault) => Err(self
                .enter_address_translation_exception_for_stage(
                    fault.addr,
                    access,
                    fault.fault,
                    fault_pc,
                    fault.host_stage,
                )),
        }
    }

    pub fn enter_address_translation_exception(
        &mut self,
        va: u64,
        access: AccessType,
        fault: TlbLookupResult,
        fault_pc: u64,
    ) -> u64 {
        self.enter_address_translation_exception_for_stage(
            va, access, fault, fault_pc, false,
        )
    }

    fn enter_address_translation_exception_for_stage(
        &mut self,
        va: u64,
        access: AccessType,
        fault: TlbLookupResult,
        fault_pc: u64,
        host_stage: bool,
    ) -> u64 {
        self.pc = fault_pc;
        if host_stage && self.in_guest_mode() {
            return self.enter_exception(
                u64::from(ECODE_GCM),
                u64::from(ESUBCODE_GCHC),
                Some(va),
            );
        }
        if self.in_guest_mode() {
            if fault == TlbLookupResult::Miss {
                let idx = CSR_TLBREHI as usize;
                self.gcsr[idx] = (self.gcsr[idx] & 0x3F) | (va & !0x1FFF);
            } else {
                self.gcsr[CSR_TLBEHI as usize] = va & !0x1FFF;
            }
        } else if fault == TlbLookupResult::Miss {
            self.tlbrehi = (self.tlbrehi & 0x3F) | (va & !0x1FFF);
        } else {
            self.tlbehi = va & !0x1FFF;
        }
        let ecode = match fault {
            TlbLookupResult::Miss => ECODE_TLBR,
            TlbLookupResult::Invalid => match access {
                AccessType::Load => ECODE_PIL,
                AccessType::Store => ECODE_PIS,
                AccessType::Fetch => ECODE_PIF,
            },
            TlbLookupResult::Dirty => ECODE_PME,
            TlbLookupResult::PrivViolation => ECODE_PPI,
            TlbLookupResult::ReadProtect => ECODE_PNR,
            TlbLookupResult::ExecProtect => ECODE_PNX,
            TlbLookupResult::Hit { .. } => unreachable!("hit is not a fault"),
        };
        self.enter_exception(u64::from(ecode), 0, Some(va))
    }

    pub(crate) fn enter_exception(
        &mut self,
        ecode: u64,
        esubcode: u64,
        badv: Option<u64>,
    ) -> u64 {
        let pc = self.pc;
        if self.in_guest_mode() && !Self::exception_exits_guest(ecode) {
            return self.enter_guest_exception(ecode, esubcode, badv, pc);
        }
        self.leave_guest_mode_for_exception();
        if ecode == u64::from(ECODE_TLBR) {
            self.tlbrera = (pc & !0x3) | 1;
            self.tlbrprmd = self.crmd & 0x7;
            if let Some(badv) = badv {
                self.tlbrbadv = badv;
            }
            self.set_crmd(
                (self.crmd & !CRMD_PLV_MASK & !CRMD_IE & !CRMD_PG) | CRMD_DA,
            );
            return self.tlbrentry;
        }

        self.era = pc;
        self.prmd = self.crmd & 0x7;
        if let Some(badv) = badv {
            self.badv = badv;
        }
        self.set_crmd(self.crmd & !CRMD_PLV_MASK & !CRMD_IE);
        self.estat = (self.estat & ESTAT_IS_MASK)
            | ((ecode & 0x3F) << 16)
            | ((esubcode & 0x1FF) << 22);
        self.exception_vector(ecode)
    }

    const fn exception_exits_guest(ecode: u64) -> bool {
        matches!(ecode as u32, ECODE_GSPR | ECODE_HVC | ECODE_GCM)
    }

    fn enter_guest_exception(
        &mut self,
        ecode: u64,
        esubcode: u64,
        badv: Option<u64>,
        pc: u64,
    ) -> u64 {
        if ecode == u64::from(ECODE_TLBR) {
            self.gcsr[CSR_TLBRERA as usize] = (pc & !0x3) | 1;
            self.gcsr[CSR_TLBRPRMD as usize] =
                self.gcsr[CSR_CRMD as usize] & 0x7;
            if let Some(badv) = badv {
                self.gcsr[CSR_TLBRBADV as usize] = badv;
            }
            self.gcsr[CSR_CRMD as usize] = (self.gcsr[CSR_CRMD as usize]
                & !CRMD_PLV_MASK
                & !CRMD_IE
                & !CRMD_PG)
                | CRMD_DA;
            return self.gcsr[CSR_TLBRENTRY as usize];
        }

        self.gcsr[CSR_ERA as usize] = pc;
        self.gcsr[CSR_PRMD as usize] = self.gcsr[CSR_CRMD as usize] & 0x7;
        if let Some(badv) = badv {
            self.gcsr[CSR_BADV as usize] = badv;
        }
        self.gcsr[CSR_CRMD as usize] &= !CRMD_PLV_MASK & !CRMD_IE;
        self.gcsr[CSR_ESTAT as usize] = (self.gcsr[CSR_ESTAT as usize]
            & ESTAT_IS_MASK)
            | ((ecode & 0x3F) << 16)
            | ((esubcode & 0x1FF) << 22);
        self.guest_exception_vector(ecode)
    }

    fn exception_vector(&self, ecode: u64) -> u64 {
        if ecode == u64::from(ECODE_TLBR) {
            return self.tlbrentry;
        }

        let vs = (self.ecfg >> ECFG_VS_SHIFT) & ECFG_VS_MASK;
        if vs == 0 {
            self.eentry
        } else {
            self.eentry
                .wrapping_add((ecode & 0x3F).wrapping_mul((1_u64 << vs) * 4))
        }
    }

    fn guest_exception_vector(&self, ecode: u64) -> u64 {
        if ecode == u64::from(ECODE_TLBR) {
            return self.gcsr[CSR_TLBRENTRY as usize];
        }

        let vs = (self.gcsr[CSR_ECFG as usize] >> ECFG_VS_SHIFT) & ECFG_VS_MASK;
        if vs == 0 {
            self.gcsr[CSR_EENTRY as usize]
        } else {
            self.gcsr[CSR_EENTRY as usize]
                .wrapping_add((ecode & 0x3F).wrapping_mul((1_u64 << vs) * 4))
        }
    }

    #[must_use]
    pub fn fast_tlb_lookup_addend(
        &self,
        va: u64,
        access: AccessType,
    ) -> Option<usize> {
        self.active_tlb_mmu().fast_tlb_lookup_addend(va, access)
    }

    pub(crate) fn flush_fast_tlb(&mut self) {
        self.mmu.flush_fast_tlb();
        self.guest_mmu.flush_fast_tlb();
    }

    pub(crate) fn invalidate_tlb_translations(&mut self) {
        self.flush_fast_tlb();
        self.request_tb_flush();
    }

    pub(crate) fn request_tb_flush(&mut self) {
        self.tb_flush_pending = true;
        self.set_exit_request();
    }

    fn fast_tlb_addend(&self, va: u64, pa: u64) -> usize {
        if pa >= self.ram_base && pa < self.ram_end {
            self.guest_base
                .wrapping_add(pa & FAST_TLB_PAGE_MASK)
                .wrapping_sub(va & FAST_TLB_PAGE_MASK) as usize
        } else {
            FAST_TLB_MMIO_ADDEND
        }
    }

    fn active_tlb_mmu(&self) -> &LoongArchMmu {
        if self.in_guest_mode() {
            &self.guest_mmu
        } else {
            &self.mmu
        }
    }

    fn active_tlb_mmu_mut(&mut self) -> &mut LoongArchMmu {
        if self.in_guest_mode() {
            &mut self.guest_mmu
        } else {
            &mut self.mmu
        }
    }

    fn active_tlbidx(&self) -> u64 {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBIDX as usize]
        } else {
            self.tlbidx
        }
    }

    fn set_active_tlbidx(&mut self, val: u64) {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBIDX as usize] = val;
        } else {
            self.tlbidx = val;
        }
    }

    fn active_tlbehi(&self) -> u64 {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBEHI as usize]
        } else {
            self.tlbehi
        }
    }

    fn set_active_tlbehi(&mut self, val: u64) {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBEHI as usize] = val;
        } else {
            self.tlbehi = val;
        }
    }

    fn set_active_tlbelo0(&mut self, val: u64) {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBELO0 as usize] = val;
        } else {
            self.tlbelo0 = val;
        }
    }

    fn set_active_tlbelo1(&mut self, val: u64) {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBELO1 as usize] = val;
        } else {
            self.tlbelo1 = val;
        }
    }

    fn active_tlbrera(&self) -> u64 {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBRERA as usize]
        } else {
            self.tlbrera
        }
    }

    fn active_tlbrbadv(&self) -> u64 {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBRBADV as usize]
        } else {
            self.tlbrbadv
        }
    }

    fn active_tlbrehi(&self) -> u64 {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBREHI as usize]
        } else {
            self.tlbrehi
        }
    }

    fn set_active_tlbrehi(&mut self, val: u64) {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBREHI as usize] = val;
        } else {
            self.tlbrehi = val;
        }
    }

    fn active_tlbrelo0(&self) -> u64 {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBRELO0 as usize]
        } else {
            self.tlbrelo0
        }
    }

    fn set_active_tlbrelo0(&mut self, val: u64) {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBRELO0 as usize] = val;
        } else {
            self.tlbrelo0 = val;
        }
    }

    fn active_tlbrelo1(&self) -> u64 {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBRELO1 as usize]
        } else {
            self.tlbrelo1
        }
    }

    fn set_active_tlbrelo1(&mut self, val: u64) {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBRELO1 as usize] = val;
        } else {
            self.tlbrelo1 = val;
        }
    }

    fn active_tlbelo0(&self) -> u64 {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBELO0 as usize]
        } else {
            self.tlbelo0
        }
    }

    fn active_tlbelo1(&self) -> u64 {
        if self.in_guest_mode() {
            self.gcsr[CSR_TLBELO1 as usize]
        } else {
            self.tlbelo1
        }
    }

    fn active_asid_low(&self) -> u16 {
        self.active_translation_asid()
    }

    fn set_active_asid_low(&mut self, asid: u64) {
        if self.in_guest_mode() {
            self.gcsr[CSR_ASID as usize] = (self.gcsr[CSR_ASID as usize]
                & !ASID_WRITE_MASK)
                | (asid & ASID_WRITE_MASK);
        } else {
            self.set_asid_low(asid);
        }
    }

    pub fn tlb_search(&self) -> Option<usize> {
        use super::mmu::{
            mtlb_flat_index, stlb_flat_index, stlb_set_index, MTLB_SIZE,
            STLB_WAYS,
        };
        let entryhi = if self.active_tlbrera() & 1 != 0 {
            self.active_tlbrehi()
        } else {
            self.active_tlbehi()
        };
        let asid = self.active_asid_low();
        let gid = self.target_gid();
        let ps = self.active_translation_stlbps();
        let mmu = self.active_tlb_mmu();
        for i in 0..MTLB_SIZE {
            let e = &mmu.mtlb[i];
            if Self::tlb_entry_matches_va(e, entryhi, asid, gid) {
                return mtlb_flat_index(i);
            }
        }
        if let Some(set_idx) = stlb_set_index(entryhi, ps) {
            for w in 0..STLB_WAYS {
                let e = &mmu.stlb[set_idx][w];
                if Self::tlb_entry_matches_va(e, entryhi, asid, gid) {
                    return stlb_flat_index(set_idx, w);
                }
            }
        }
        None
    }

    pub fn tlb_search_and_update(&mut self) {
        if let Some(idx) = self.tlb_search() {
            let tlbidx = self.active_tlbidx();
            self.set_active_tlbidx(
                (tlbidx & !(0xFFF | (1 << 31))) | idx as u64,
            );
        } else {
            self.set_active_tlbidx(self.active_tlbidx() | (1 << 31));
        }
    }

    fn tlb_entry_matches_va(
        entry: &super::mmu::TlbEntry,
        va: u64,
        asid: u16,
        gid: u8,
    ) -> bool {
        if !entry.valid || entry.gid != gid || (!entry.g && entry.asid != asid)
        {
            return false;
        }
        let ps = u32::from(entry.page_size);
        if ps >= 63 {
            return false;
        }
        let pair_mask = !((1_u64 << (ps + 1)) - 1);
        (va & super::mmu::TARGET_VIRT_MASK & pair_mask)
            == ((entry.vppn << 13) & super::mmu::TARGET_VIRT_MASK & pair_mask)
    }

    fn page_walk_dir_base_width(&self, level: u64) -> (u64, u64) {
        let pwcl = if self.in_guest_mode() {
            self.gcsr[CSR_PWCL as usize]
        } else {
            self.pwcl
        };
        let pwch = if self.in_guest_mode() {
            self.gcsr[CSR_PWCH as usize]
        } else {
            self.pwch
        };
        match level {
            1 => ((pwcl >> 10) & 0x1F, (pwcl >> 15) & 0x1F),
            2 => ((pwcl >> 20) & 0x1F, (pwcl >> 25) & 0x1F),
            3 => (pwch & 0x3F, (pwch >> 6) & 0x3F),
            4 => ((pwch >> 12) & 0x3F, (pwch >> 18) & 0x3F),
            _ => (pwcl & 0x1F, (pwcl >> 5) & 0x1F),
        }
    }

    fn read_phys_u64(&self, pa: u64) -> u64 {
        if self.guest_base == 0 {
            return 0;
        }
        if pa < self.ram_base
            || pa.checked_add(8).is_none_or(|end| end > self.ram_end)
        {
            return 0;
        }
        let ptr = (self.guest_base + pa) as *const u64;
        // SAFETY: bounds were checked against the RAM window above.
        unsafe { ptr.read_unaligned() }
    }

    fn read_page_walk_u64(&self, gpa: u64) -> u64 {
        let pa = if self.in_guest_mode() {
            match self.translate_guest_host_stage(gpa, AccessType::Load) {
                TlbLookupResult::Hit { pa, .. } => pa,
                _ => return 0,
            }
        } else {
            gpa
        };
        self.read_phys_u64(pa)
    }

    fn sanitize_page_walk_pte(&self, pte: u64) -> u64 {
        pte & HW_PTE_MASK_LA64
    }

    pub fn lddir(&self, base: u64, level: u64) -> u64 {
        if level == 0 || level > 4 {
            return base;
        }

        if base & TLBENTRY_HUGE != 0 {
            if level == 4
                || ((base >> TLBENTRY_LEVEL_SHIFT) & TLBENTRY_LEVEL_MASK) != 0
            {
                return base;
            }
            return base
                | ((level & TLBENTRY_LEVEL_MASK) << TLBENTRY_LEVEL_SHIFT);
        }

        let badv = self.active_tlbrbadv();
        let (dir_base, dir_width) = self.page_walk_dir_base_width(level);
        if dir_width == 0 || dir_width >= 64 {
            return 0;
        }
        let index = (badv >> dir_base) & ((1_u64 << dir_width) - 1);
        let phys = (base & super::mmu::TARGET_VIRT_MASK) | (index << 3);
        self.read_page_walk_u64(phys) & super::mmu::TARGET_VIRT_MASK
    }

    pub fn ldpte(&mut self, base: u64, odd: u64) {
        let tmp;
        let ps;
        if base & TLBENTRY_HUGE != 0 {
            let level = (base >> TLBENTRY_LEVEL_SHIFT) & TLBENTRY_LEVEL_MASK;
            let (dir_base, dir_width) = self.page_walk_dir_base_width(level);
            let mut huge = base & !TLBENTRY_HUGE;
            huge &= !(TLBENTRY_LEVEL_MASK << TLBENTRY_LEVEL_SHIFT);
            if huge & TLBENTRY_HGLOBAL != 0 {
                huge &= !TLBENTRY_HGLOBAL;
                huge |= 1 << 6;
            }
            ps = dir_base + dir_width - 1;
            tmp = self.sanitize_page_walk_pte(if odd != 0 {
                huge.wrapping_add(1_u64 << ps)
            } else {
                huge
            });
        } else {
            let pwcl = if self.in_guest_mode() {
                self.gcsr[CSR_PWCL as usize]
            } else {
                self.pwcl
            };
            let ptbase = pwcl & 0x1F;
            let ptwidth = (pwcl >> 5) & 0x1F;
            if ptwidth == 0 || ptwidth >= 64 {
                return;
            }
            let badv = self.active_tlbrbadv();
            let ptindex = ((badv >> ptbase) & ((1_u64 << ptwidth) - 1)) & !1;
            let offset = if odd != 0 { ptindex + 1 } else { ptindex } << 3;
            let phys = (base & super::mmu::TARGET_VIRT_MASK) | offset;
            tmp = self.sanitize_page_walk_pte(self.read_page_walk_u64(phys));
            ps = ptbase;
        }

        if odd != 0 {
            self.set_active_tlbrelo1(tmp);
        } else {
            self.set_active_tlbrelo0(tmp);
        }
        self.set_active_tlbrehi((self.active_tlbrehi() & !0x3F) | (ps & 0x3F));
    }

    pub fn tlb_read(&mut self, idx: usize) {
        let Some(e) = self.active_tlb_mmu().entry(idx).copied() else {
            self.read_invalid_tlb_entry();
            return;
        };
        if !e.valid || (self.in_guest_mode() && e.gid != self.target_gid()) {
            self.read_invalid_tlb_entry();
            return;
        }
        let tlbidx = self.active_tlbidx();
        self.set_active_tlbidx(
            (tlbidx & !(1 << 31 | 0x3F << 24 | 0xFFF))
                | ((u64::from(e.page_size)) << 24)
                | (idx as u64 & 0xFFF),
        );
        self.set_active_tlbehi(e.vppn << 13);
        let tlbelo0 = self.encode_tlbelo(
            e.ppn0, e.v0, e.d0, e.plv0, e.mat0, e.g, e.nr0, e.nx0, e.rplv0,
        );
        let tlbelo1 = self.encode_tlbelo(
            e.ppn1, e.v1, e.d1, e.plv1, e.mat1, e.g, e.nr1, e.nx1, e.rplv1,
        );
        self.set_active_tlbelo0(tlbelo0);
        self.set_active_tlbelo1(tlbelo1);
        self.set_active_asid_low(u64::from(e.asid));
    }

    fn read_invalid_tlb_entry(&mut self) {
        self.set_active_tlbidx(
            (self.active_tlbidx() & !(0x3F << 24)) | (1 << 31),
        );
        self.set_active_tlbehi(0);
        self.set_active_tlbelo0(0);
        self.set_active_tlbelo1(0);
        self.set_active_asid_low(0);
    }

    pub fn tlb_write(&mut self, idx: usize) {
        if self.active_tlbidx() & (1 << 31) != 0 {
            if self
                .active_tlb_mmu_mut()
                .write_entry(idx, super::mmu::TlbEntry::default())
            {
                self.invalidate_tlb_translations();
            }
            return;
        }
        let (entryhi, elo0, elo1, ps) = self.tlb_entry_source_csrs();
        let entry = self.tlb_entry_from_csrs(entryhi, elo0, elo1, ps);
        if self.active_tlb_mmu_mut().write_entry(idx, entry) {
            self.invalidate_tlb_translations();
        }
    }

    fn tlb_entry_source_csrs(&self) -> (u64, u64, u64, u8) {
        if self.active_tlbrera() & 1 != 0 {
            (
                self.active_tlbrehi(),
                self.active_tlbrelo0(),
                self.active_tlbrelo1(),
                (self.active_tlbrehi() & 0x3F) as u8,
            )
        } else {
            (
                self.active_tlbehi(),
                self.active_tlbelo0(),
                self.active_tlbelo1(),
                ((self.active_tlbidx() >> 24) & 0x3F) as u8,
            )
        }
    }

    fn tlb_entry_from_csrs(
        &self,
        entryhi: u64,
        elo0: u64,
        elo1: u64,
        ps: u8,
    ) -> super::mmu::TlbEntry {
        super::mmu::TlbEntry {
            vppn: entryhi >> 13,
            page_size: ps,
            asid: self.active_asid_low(),
            gid: self.target_gid(),
            g: (elo0 & elo1 & (1 << 6)) != 0,
            valid: true,
            ppn0: (elo0 >> 12) & 0xF_FFFF_FFFF,
            ppn1: (elo1 >> 12) & 0xF_FFFF_FFFF,
            plv0: ((elo0 >> 2) & 0x3) as u8,
            plv1: ((elo1 >> 2) & 0x3) as u8,
            mat0: ((elo0 >> 4) & 0x3) as u8,
            mat1: ((elo1 >> 4) & 0x3) as u8,
            d0: elo0 & (1 << 1) != 0,
            d1: elo1 & (1 << 1) != 0,
            v0: elo0 & 1 != 0,
            v1: elo1 & 1 != 0,
            nr0: elo0 & (1 << 61) != 0,
            nr1: elo1 & (1 << 61) != 0,
            nx0: elo0 & (1 << 62) != 0,
            nx1: elo1 & (1 << 62) != 0,
            rplv0: elo0 & (1 << 63) != 0,
            rplv1: elo1 & (1 << 63) != 0,
        }
    }

    pub fn tlb_fill(&mut self) {
        use super::mmu::{
            mtlb_flat_index, stlb_flat_index, stlb_set_index, MTLB_SIZE,
        };
        let (entryhi, elo0, elo1, ps) = self.tlb_entry_source_csrs();
        let entry = self.tlb_entry_from_csrs(entryhi, elo0, elo1, ps);
        let idx = if ps == self.active_translation_stlbps() {
            stlb_set_index(entryhi, ps).and_then(|set| stlb_flat_index(set, 0))
        } else {
            let raw_idx = (self.active_tlbidx() as usize) & (MTLB_SIZE - 1);
            mtlb_flat_index(raw_idx)
        };
        if let Some(idx) = idx {
            if self.active_tlb_mmu_mut().write_entry(idx, entry) {
                self.invalidate_tlb_translations();
            }
        }
    }

    pub fn tlb_clear_by_index(&mut self) {
        use super::mmu::{STLB_SETS, STLB_SIZE, TLB_TOTAL_SIZE};

        let idx = (self.active_tlbidx() & 0xFFF) as usize;
        let asid = self.active_asid_low();
        let gid = self.target_gid();
        let mmu = self.active_tlb_mmu_mut();
        if idx < STLB_SIZE {
            let set = idx % STLB_SETS;
            mmu.stlb[set].iter_mut().for_each(|e| {
                if e.valid && e.gid == gid && !e.g && e.asid == asid {
                    e.valid = false;
                }
            });
        } else if idx < TLB_TOTAL_SIZE {
            mmu.mtlb.iter_mut().for_each(|e| {
                if e.valid && e.gid == gid && !e.g && e.asid == asid {
                    e.valid = false;
                }
            });
        }
        self.invalidate_tlb_translations();
    }

    pub fn tlb_flush_by_index(&mut self) {
        use super::mmu::{STLB_SETS, STLB_SIZE, TLB_TOTAL_SIZE};

        let idx = (self.active_tlbidx() & 0xFFF) as usize;
        let mmu = self.active_tlb_mmu_mut();
        if idx < STLB_SIZE {
            let set = idx % STLB_SETS;
            mmu.stlb[set].iter_mut().for_each(|e| {
                e.valid = false;
            });
        } else if idx < TLB_TOTAL_SIZE {
            mmu.mtlb.iter_mut().for_each(|e| {
                e.valid = false;
            });
        }
        self.invalidate_tlb_translations();
    }

    pub fn invtlb(&mut self, op: u32, asid: u16, va: u64) {
        let should_flush = matches!(op, 0..=6 | 0x9..=0xE | 0x10..=0x16);
        let gid = self.target_gid();
        match op {
            0 | 1 => {
                let mmu = self.active_tlb_mmu_mut();
                mmu.mtlb.iter_mut().for_each(|e| e.valid = false);
                mmu.stlb.iter_mut().for_each(|s| {
                    s.iter_mut().for_each(|e| e.valid = false);
                });
            }
            2 => {
                let mmu = self.active_tlb_mmu_mut();
                mmu.mtlb.iter_mut().for_each(|e| {
                    if e.g && e.gid == gid {
                        e.valid = false;
                    }
                });
                mmu.stlb.iter_mut().for_each(|s| {
                    s.iter_mut().for_each(|e| {
                        if e.g && e.gid == gid {
                            e.valid = false;
                        }
                    });
                });
            }
            3 => {
                let mmu = self.active_tlb_mmu_mut();
                mmu.mtlb.iter_mut().for_each(|e| {
                    if !e.g && e.gid == gid {
                        e.valid = false;
                    }
                });
                mmu.stlb.iter_mut().for_each(|s| {
                    s.iter_mut().for_each(|e| {
                        if !e.g && e.gid == gid {
                            e.valid = false;
                        }
                    });
                });
            }
            4 => {
                let mmu = self.active_tlb_mmu_mut();
                mmu.mtlb.iter_mut().for_each(|e| {
                    if !e.g && e.gid == gid && e.asid == asid {
                        e.valid = false;
                    }
                });
                mmu.stlb.iter_mut().for_each(|s| {
                    s.iter_mut().for_each(|e| {
                        if !e.g && e.gid == gid && e.asid == asid {
                            e.valid = false;
                        }
                    });
                });
            }
            5 => {
                let mmu = self.active_tlb_mmu_mut();
                mmu.mtlb.iter_mut().for_each(|e| {
                    if !e.g && Self::tlb_entry_matches_va(e, va, asid, gid) {
                        e.valid = false;
                    }
                });
                mmu.stlb.iter_mut().for_each(|s| {
                    s.iter_mut().for_each(|e| {
                        if !e.g && Self::tlb_entry_matches_va(e, va, asid, gid)
                        {
                            e.valid = false;
                        }
                    });
                });
            }
            6 => {
                let mmu = self.active_tlb_mmu_mut();
                mmu.mtlb.iter_mut().for_each(|e| {
                    if Self::tlb_entry_matches_va(e, va, asid, gid) {
                        e.valid = false;
                    }
                });
                mmu.stlb.iter_mut().for_each(|s| {
                    s.iter_mut().for_each(|e| {
                        if Self::tlb_entry_matches_va(e, va, asid, gid) {
                            e.valid = false;
                        }
                    });
                });
            }
            0x9..=0xE | 0x10..=0x16 => {
                self.mmu.mtlb.iter_mut().for_each(|e| e.valid = false);
                self.mmu.stlb.iter_mut().for_each(|s| {
                    s.iter_mut().for_each(|e| e.valid = false);
                });
                self.guest_mmu.mtlb.iter_mut().for_each(|e| e.valid = false);
                self.guest_mmu.stlb.iter_mut().for_each(|s| {
                    s.iter_mut().for_each(|e| e.valid = false);
                });
            }
            _ => {}
        }
        if should_flush {
            self.invalidate_tlb_translations();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_tlbelo(
        &self,
        ppn: u64,
        v: bool,
        d: bool,
        plv: u8,
        mat: u8,
        g: bool,
        nr: bool,
        nx: bool,
        rplv: bool,
    ) -> u64 {
        (u64::from(v))
            | (u64::from(d) << 1)
            | (u64::from(plv) << 2)
            | (u64::from(mat) << 4)
            | (u64::from(g) << 6)
            | (ppn << 12)
            | (u64::from(nr) << 61)
            | (u64::from(nx) << 62)
            | (u64::from(rplv) << 63)
    }

    pub fn env_ptr(&mut self) -> *mut u8 {
        std::ptr::from_mut(self).cast()
    }

    pub fn set_estat_hw(&mut self, val: u64) {
        let was_pending = self.pending_interrupt();
        self.estat = val;
        self.wake_if_new_enabled_interrupt(was_pending);
    }

    #[must_use]
    pub fn is_halted(&self) -> bool {
        self.halted.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn set_halted_flag(&mut self, val: bool) {
        self.halted.store(val, std::sync::atomic::Ordering::Release);
    }

    pub fn take_tb_flush(&mut self) -> bool {
        let p = self.tb_flush_pending;
        self.tb_flush_pending = false;
        p
    }

    pub fn set_translation_fault_pending(&mut self) {
        self.translation_fault_pending = true;
        self.mem_fault_cause = 1;
        self.fault_pc = self.pc;
    }

    pub fn set_memory_fault_pending(&mut self, fault_pc: u64) {
        self.mem_fault_cause = 1;
        self.fault_pc = fault_pc;
    }

    pub fn take_translation_fault_pending(&mut self) -> bool {
        let pending =
            self.translation_fault_pending || self.mem_fault_cause != 0;
        self.translation_fault_pending = false;
        self.mem_fault_cause = 0;
        self.fault_pc = 0;
        pending
    }

    #[must_use]
    pub fn guest_base_val(&self) -> u64 {
        self.guest_base
    }

    pub fn set_guest_base(&mut self, val: u64) {
        self.guest_base = val;
    }

    #[must_use]
    pub fn ram_base_val(&self) -> u64 {
        self.ram_base
    }

    pub fn set_ram_base(&mut self, val: u64) {
        self.ram_base = val;
    }

    #[must_use]
    pub fn ram_end_val(&self) -> u64 {
        self.ram_end
    }

    pub fn set_ram_end(&mut self, val: u64) {
        self.ram_end = val;
    }

    #[must_use]
    pub fn address_space_ptr(&self) -> u64 {
        self.as_ptr
    }

    pub fn set_address_space_ptr(&mut self, ptr: u64) {
        self.as_ptr = ptr;
    }

    #[must_use]
    pub fn fault_pc_val(&self) -> u64 {
        self.fault_pc
    }

    pub fn set_exit_request(&mut self) {
        self.neg_align = -1;
    }

    pub fn reset_exit_request(&mut self) {
        self.neg_align = 0;
    }

    #[must_use]
    pub fn neg_align_val(&self) -> i32 {
        self.neg_align
    }

    pub fn set_last_phys_pc(&mut self, val: u64) {
        self.last_phys_pc = val;
    }

    #[must_use]
    pub fn last_phys_pc_val(&self) -> u64 {
        self.last_phys_pc
    }

    pub fn set_badv_raw(&mut self, val: u64) {
        self.badv = val;
    }

    #[must_use]
    pub fn tval(&self) -> u64 {
        self.tval
    }

    pub fn iocsr_read(&self, addr: u32, width: u32) -> u64 {
        if let Some(val) =
            self.iocsr_dispatcher.read(self.cpuid as u32, addr, width)
        {
            return mask_iocsr_width(val, width);
        }
        self.local_iocsr_read(addr, width)
    }

    fn local_iocsr_read(&self, addr: u32, width: u32) -> u64 {
        let mask: u64 = match width {
            1 => 0xFF,
            2 => 0xFFFF,
            4 => 0xFFFF_FFFF,
            _ => u64::MAX,
        };
        let (reg_val, byte_off) = self.iocsr_reg_read(addr);
        let shift = u64::from(byte_off) * 8;
        (reg_val >> shift) & mask
    }

    fn iocsr_reg_read(&self, addr: u32) -> (u64, u32) {
        let base = addr & !0x7;
        let off = addr & 0x7;
        let val = match base {
            0x1000 => self.ipi_status | (self.ipi_enable << 32),
            0x1008 => 0,
            0x1020 => self.ipi_mailbox[0],
            0x1028 => self.ipi_mailbox[1],
            0x1030 => self.ipi_mailbox[2],
            0x1038 => self.ipi_mailbox[3],
            _ => 0,
        };
        (val, off)
    }

    pub fn iocsr_write(&mut self, addr: u32, val: u64, width: u32) {
        if self
            .iocsr_dispatcher
            .write(self.cpuid as u32, addr, width, val)
        {
            return;
        }
        self.local_iocsr_write(addr, val, width);
    }

    fn local_iocsr_write(&mut self, addr: u32, val: u64, width: u32) {
        let mask: u64 = match width {
            1 => 0xFF,
            2 => 0xFFFF,
            4 => 0xFFFF_FFFF,
            _ => u64::MAX,
        };
        let masked_val = val & mask;
        match addr {
            a if (0x1000..0x1004).contains(&a) => {
                // Status: read-only, writes ignored
            }
            a if (0x1004..0x1008).contains(&a) => {
                let off = a - 0x1004;
                let shift = u64::from(off) * 8;
                let wmask = mask << shift;
                let wval = masked_val << shift;
                self.ipi_enable = (self.ipi_enable & !wmask) | wval;
                self.update_ipi_interrupt();
            }
            a if (0x1008..0x100C).contains(&a) => {
                let off = a - 0x1008;
                let shift = u64::from(off) * 8;
                self.ipi_status |= masked_val << shift;
                self.update_ipi_interrupt();
            }
            a if (0x100C..0x1010).contains(&a) => {
                let off = a - 0x100C;
                let shift = u64::from(off) * 8;
                self.ipi_status &= !(masked_val << shift);
                self.update_ipi_interrupt();
            }
            0x1040 => {
                let target = (masked_val >> 16) & 0x3FF;
                let vector = (masked_val & 0x1F) as u32;
                if target == 0 {
                    self.ipi_status |= 1u64 << vector;
                    self.update_ipi_interrupt();
                }
            }
            0x1048 => {
                let dest = 0x1020 + (masked_val as u32 & 0x1C);
                self.send_ipi_data(dest, masked_val);
            }
            0x1158 => {
                let dest = (masked_val & 0xFFFF) as u32;
                self.send_ipi_data(dest, masked_val);
            }
            a if (0x1020..0x1040).contains(&a) => {
                let mb_off = a - 0x1020;
                let mb_idx = (mb_off / 8) as usize;
                let byte_off = mb_off % 8;
                if mb_idx < 4 {
                    let shift = u64::from(byte_off) * 8;
                    let wmask = mask << shift;
                    let wval = masked_val << shift;
                    let old = self.ipi_mailbox[mb_idx];
                    self.ipi_mailbox[mb_idx] = (old & !wmask) | wval;
                }
            }
            _ => {}
        }
    }

    fn send_ipi_data(&mut self, dest_addr: u32, val: u64) {
        let target = (val >> 16) & 0x3FF;
        if target != 0 {
            return;
        }
        let data = (val >> 32) as u32;
        let byte_mask = ((val >> 27) & 0xF) as u32;
        let merged = self.merge_ipi_word(dest_addr, data, byte_mask);
        self.write_local_iocsr_word(dest_addr, merged);
    }

    fn merge_ipi_word(&self, dest_addr: u32, data: u32, byte_mask: u32) -> u32 {
        let old = self.read_local_iocsr_word(dest_addr);
        let mut word = old;
        for b in 0..4u32 {
            if byte_mask & (1 << b) == 0 {
                let s = b * 8;
                word = (word & !(0xFF << s)) | ((data >> s & 0xFF) << s);
            }
        }
        word
    }

    fn read_local_iocsr_word(&self, addr: u32) -> u32 {
        match addr {
            a if (0x1000..0x1004).contains(&a) => self.ipi_status as u32,
            a if (0x1004..0x1008).contains(&a) => self.ipi_enable as u32,
            a if (0x1020..0x1040).contains(&a) => {
                let off = a - 0x1020;
                let idx = (off / 8) as usize;
                let hi = (off / 4) & 1 != 0;
                if idx < 4 {
                    let shift = if hi { 32 } else { 0 };
                    ((self.ipi_mailbox[idx] >> shift) & 0xFFFF_FFFF) as u32
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    fn write_local_iocsr_word(&mut self, addr: u32, val: u32) {
        match addr {
            a if (0x1000..0x1004).contains(&a) => {}
            a if (0x1004..0x1008).contains(&a) => {
                self.ipi_enable = u64::from(val);
                self.update_ipi_interrupt();
            }
            a if (0x1008..0x100C).contains(&a) => {
                self.ipi_status |= u64::from(val);
                self.update_ipi_interrupt();
            }
            a if (0x100C..0x1010).contains(&a) => {
                self.ipi_status &= !u64::from(val);
                self.update_ipi_interrupt();
            }
            a if (0x1020..0x1040).contains(&a) => {
                let off = a - 0x1020;
                let idx = (off / 8) as usize;
                let hi = (off / 4) & 1 != 0;
                if idx < 4 {
                    let shift = if hi { 32 } else { 0 };
                    let cleared =
                        self.ipi_mailbox[idx] & !(0xFFFF_FFFF_u64 << shift);
                    self.ipi_mailbox[idx] = cleared | (u64::from(val) << shift);
                }
            }
            0x1040 => {
                let target = (val >> 16) & 0x3FF;
                if target == 0 {
                    let vector = val & 0x1F;
                    self.ipi_status |= 1u64 << vector;
                    self.update_ipi_interrupt();
                }
            }
            _ => {}
        }
    }

    fn update_ipi_interrupt(&mut self) {
        let was_pending = self.estat & (1 << 12) != 0;
        if self.ipi_status & self.ipi_enable != 0 {
            self.estat |= 1 << 12;
            if !was_pending {
                self.wake_for_interrupt_request();
            }
        } else {
            self.estat &= !(1 << 12);
        }
    }

    pub fn set_hwi_interrupt_pending(&mut self, hwi: u8, pending: bool) {
        if hwi >= 8 {
            return;
        }
        let bit = 2 + u32::from(hwi);
        if self.gintc & (1_u64 << (8 + hwi)) != 0 {
            self.set_guest_interrupt_bit_pending(bit, pending);
        } else {
            self.set_interrupt_bit_pending(bit, pending);
        }
    }

    pub fn set_ipi_interrupt_pending(&mut self, pending: bool) {
        self.set_interrupt_bit_pending(12, pending);
    }

    pub(crate) fn set_timer_interrupt_pending(&mut self, pending: bool) {
        self.set_interrupt_bit_pending(11, pending);
    }

    fn set_interrupt_bit_pending(&mut self, bit: u32, pending: bool) {
        let mask = 1_u64 << bit;
        let was_pending = self.estat & mask != 0;
        if pending {
            self.estat |= mask;
            if !was_pending {
                self.wake_for_interrupt_request();
            }
        } else {
            self.estat &= !mask;
        }
    }

    pub(crate) fn set_guest_interrupt_bit_pending(
        &mut self,
        bit: u32,
        pending: bool,
    ) {
        let mask = 1_u64 << bit;
        let was_pending = self.gcsr[CSR_ESTAT as usize] & mask != 0;
        if pending {
            self.gcsr[CSR_ESTAT as usize] |= mask;
            if !was_pending {
                self.wake_for_interrupt_request();
            }
        } else {
            self.gcsr[CSR_ESTAT as usize] &= !mask;
        }
    }

    pub(crate) fn wake_if_new_enabled_interrupt(&mut self, was_pending: bool) {
        if !was_pending && self.pending_interrupt() {
            self.wake_for_interrupt_request();
        }
    }

    fn wake_for_interrupt_request(&mut self) {
        self.halted
            .store(false, std::sync::atomic::Ordering::Release);
        self.set_exit_request();
    }
}

impl LoongArchCpu {
    fn tick_timer_values(tcfg: &mut u64, tval: &mut u64, cycles: u64) -> bool {
        if *tcfg & 1 == 0 {
            return false;
        }
        if *tval == 0 {
            return false;
        }
        if *tval > cycles {
            *tval -= cycles;
            return false;
        }

        let periodic = *tcfg & 2 != 0;
        if periodic {
            *tval = *tcfg & !0x3;
        } else {
            *tval = 0;
            *tcfg &= !1;
        }
        true
    }

    pub fn timer_tick(&mut self, cycles: u64) {
        if Self::tick_timer_values(&mut self.tcfg, &mut self.tval, cycles) {
            self.set_timer_interrupt_pending(true);
        }
        if self.in_guest_mode() {
            let mut tcfg = self.gcsr[CSR_TCFG as usize];
            let mut tval = self.gcsr[CSR_TVAL as usize];
            let expired = Self::tick_timer_values(&mut tcfg, &mut tval, cycles);
            self.gcsr[CSR_TCFG as usize] = tcfg;
            self.gcsr[CSR_TVAL as usize] = tval;
            if expired {
                self.set_guest_interrupt_bit_pending(11, true);
            }
        }
    }
}

impl Default for LoongArchCpu {
    fn default() -> Self {
        Self::new()
    }
}

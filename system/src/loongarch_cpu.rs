use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use machina_accel::ir::context::Context;
use machina_accel::ir::tb::{EXCP_LOONGARCH_DONE, EXCP_LOONGARCH_WFI};
use machina_accel::x86_64::emitter::SoftMmuConfig;
use machina_accel::{ArchExitAction, GuestCpu};
use machina_core::address::GPA;
use machina_guest_loongarch::loongarch::cpu::{
    LoongArchCpu, LoongArchCpuInterruptState, FAST_TLB_PTR_OFFSET,
    FAULT_PC_OFFSET, MEM_FAULT_CAUSE_OFFSET,
};
use machina_guest_loongarch::loongarch::csr::*;
use machina_guest_loongarch::loongarch::ext::LoongArchCfg;
use machina_guest_loongarch::loongarch::mmu;
use machina_guest_loongarch::loongarch::trans::{
    LoongArchDisasContext, LoongArchTranslator,
};
use machina_guest_loongarch::{DisasJumpType, TranslatorOps};
use machina_memory::address_space::AddressSpace;

pub const LOONGARCH_TB_FLAG_PLV_MASK: u32 = 0b0000_0011;
pub const LOONGARCH_TB_FLAG_DA: u32 = 1 << 2;
pub const LOONGARCH_TB_FLAG_PG: u32 = 1 << 3;
pub const LOONGARCH_TB_FLAG_FPE: u32 = 1 << 4;

const TARGET_PAGE_SIZE: u64 = 0x1000;
const TARGET_PAGE_MASK: u64 = !(TARGET_PAGE_SIZE - 1);
const TARGET_PAGE_OFFSET_MASK: u64 = TARGET_PAGE_SIZE - 1;
const LOONGARCH_MAX_TB_INSNS: u32 = 64;

#[must_use]
pub fn loongarch_soft_mmu_config() -> SoftMmuConfig {
    SoftMmuConfig {
        tlb_ptr_offset: FAST_TLB_PTR_OFFSET,
        entry_size: mmu::fast_tlb_offsets::ENTRY_SIZE,
        addr_read_off: mmu::fast_tlb_offsets::ADDR_READ,
        addr_write_off: mmu::fast_tlb_offsets::ADDR_WRITE,
        addend_off: mmu::fast_tlb_offsets::ADDEND,
        index_mask: (mmu::FAST_TLB_SIZE - 1) as u64,
        load_helper: loongarch_mem_read as *const () as u64,
        store_helper: loongarch_mem_write as *const () as u64,
        fault_cause_offset: MEM_FAULT_CAUSE_OFFSET,
        fault_pc_offset: FAULT_PC_OFFSET,
        dirty_offset: mmu::fast_tlb_offsets::DIRTY,
        tb_ret_addr: 0,
    }
}

pub struct LoongArchFullSystemCpu {
    pub cpu: LoongArchCpu,
    stop_flag: Arc<AtomicBool>,
    run_gate: Option<Arc<AtomicBool>>,
    interrupts: Option<Arc<LoongArchCpuInterruptState>>,
}

unsafe impl Send for LoongArchFullSystemCpu {}

impl LoongArchFullSystemCpu {
    /// # Safety
    /// `ram_ptr` must point to valid memory of `ram_size` bytes.
    pub unsafe fn new(
        mut cpu: LoongArchCpu,
        ram_ptr: *const u8,
        ram_base: u64,
        ram_size: u64,
        address_space_ptr: u64,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        configure_cpu_memory(
            &mut cpu,
            ram_ptr,
            ram_base,
            ram_size,
            address_space_ptr,
        );
        Self {
            cpu,
            stop_flag,
            run_gate: None,
            interrupts: None,
        }
    }

    /// # Safety
    /// `ram_ptr` must point to valid memory of `ram_size` bytes.
    pub unsafe fn new_with_interrupts(
        mut cpu: LoongArchCpu,
        ram_ptr: *const u8,
        ram_base: u64,
        ram_size: u64,
        address_space_ptr: u64,
        stop_flag: Arc<AtomicBool>,
        interrupts: Arc<LoongArchCpuInterruptState>,
    ) -> Self {
        configure_cpu_memory(
            &mut cpu,
            ram_ptr,
            ram_base,
            ram_size,
            address_space_ptr,
        );
        Self {
            cpu,
            stop_flag,
            run_gate: None,
            interrupts: Some(interrupts),
        }
    }

    pub fn set_run_gate(&mut self, run_gate: Option<Arc<AtomicBool>>) {
        self.run_gate = run_gate;
    }

    fn run_gate_open(&self) -> bool {
        self.run_gate
            .as_ref()
            .is_none_or(|gate| gate.load(Ordering::Acquire))
    }

    fn apply_async_interrupts(&mut self) {
        if let Some(interrupts) = &self.interrupts {
            interrupts.apply_to_cpu(&mut self.cpu);
        }
    }

    fn async_pending_interrupt(&self) -> bool {
        self.interrupts
            .as_ref()
            .is_some_and(|interrupts| interrupts.pending_interrupt(&self.cpu))
    }

    fn async_has_pending_irq(&self) -> bool {
        self.interrupts
            .as_ref()
            .is_some_and(|interrupts| interrupts.has_pending_irq())
    }

    pub fn wake_waiters(&self) {
        if let Some(interrupts) = &self.interrupts {
            interrupts.wake_waiters();
        }
    }

    #[must_use]
    pub fn interrupt_state(&self) -> Option<Arc<LoongArchCpuInterruptState>> {
        self.interrupts.as_ref().map(Arc::clone)
    }

    fn handle_wfi(&mut self) -> ArchExitAction {
        self.cpu.set_halted_flag(true);
        if !self.cpu.pending_interrupt() {
            let tcfg = self.cpu.csr_read(CSR_TCFG);
            let tval = self.cpu.tval();
            if tcfg & 1 != 0 && tval != 0 {
                self.cpu.timer_tick(tval);
            }
        }
        if !self.cpu.pending_interrupt() {
            if !self.wait_for_interrupt() {
                self.cpu.set_halted_flag(false);
                return ArchExitAction::Halted;
            }
            self.apply_async_interrupts();
        }
        self.cpu.set_halted_flag(false);
        if self.cpu.pending_interrupt() {
            loongarch_handle_interrupt(&mut self.cpu);
        }
        ArchExitAction::Continue
    }
}

impl GuestCpu for LoongArchFullSystemCpu {
    type IrContext = Context;

    fn get_pc(&self) -> u64 {
        self.cpu.pc()
    }

    fn get_flags(&self) -> u32 {
        loongarch_tb_flags(&self.cpu)
    }

    fn gen_code(&mut self, ir: &mut Context, pc: u64, max_insns: u32) -> u32 {
        self.apply_async_interrupts();
        loongarch_gen_code(&mut self.cpu, ir, pc, max_insns)
    }

    fn env_ptr(&mut self) -> *mut u8 {
        self.cpu.env_ptr()
    }

    fn pending_interrupt(&self) -> bool {
        self.cpu.pending_interrupt() || self.async_pending_interrupt()
    }

    fn has_pending_irq(&self) -> bool {
        self.cpu.masked_interrupt_line().is_some()
            || self.async_has_pending_irq()
            || loongarch_timer_counting(&self.cpu)
    }

    fn is_halted(&self) -> bool {
        self.cpu.is_halted()
    }

    fn set_halted(&mut self, halted: bool) {
        self.cpu.set_halted_flag(halted);
    }

    fn set_pc(&mut self, pc: u64) {
        self.apply_async_interrupts();
        self.cpu.set_pc(pc);
    }

    fn handle_interrupt(&mut self) {
        self.apply_async_interrupts();
        loongarch_handle_interrupt(&mut self.cpu);
    }

    fn handle_exception(&mut self, _cause: u64, _tval: u64) {
        loongarch_handle_exception(&mut self.cpu);
    }

    fn handle_arch_exit(&mut self, code: u64) -> ArchExitAction {
        self.apply_async_interrupts();
        match code {
            EXCP_LOONGARCH_DONE => ArchExitAction::Continue,
            EXCP_LOONGARCH_WFI => self.handle_wfi(),
            _ => ArchExitAction::Exit(code as usize),
        }
    }

    fn check_mem_fault(&mut self) -> bool {
        self.apply_async_interrupts();
        self.cpu.take_translation_fault_pending()
    }

    fn on_tb_executed(&mut self, _guest_size: u32, guest_insns: u16) {
        self.apply_async_interrupts();
        let insns = u32::from(guest_insns).max(1);
        self.cpu.timer_tick(u64::from(insns));
    }

    fn set_exit_request(&mut self) {
        self.cpu.set_exit_request();
    }

    fn reset_exit_request(&mut self) {
        self.apply_async_interrupts();
        if self.cpu.masked_interrupt_line().is_some()
            || self.async_has_pending_irq()
            || loongarch_timer_counting(&self.cpu)
        {
            self.cpu.set_exit_request();
        } else {
            self.cpu.reset_exit_request();
        }
    }

    fn last_phys_pc(&self) -> u64 {
        self.cpu.last_phys_pc_val()
    }

    fn translate_pc(&self, vpc: u64) -> u64 {
        match self.cpu.translate_address(vpc, mmu::AccessType::Fetch) {
            mmu::TlbLookupResult::Hit { pa, .. } => pa,
            _ => u64::MAX,
        }
    }

    fn should_exit(&self) -> bool {
        !self.stop_flag.load(Ordering::Relaxed) || !self.run_gate_open()
    }

    fn wait_for_interrupt(&self) -> bool {
        self.interrupts.as_ref().is_some_and(|interrupts| {
            interrupts
                .wait_for_irq_or_stop(&self.stop_flag, self.run_gate.as_deref())
        })
    }

    fn take_tb_flush_pending(&mut self) -> bool {
        self.apply_async_interrupts();
        self.cpu.take_tb_flush()
    }
}

fn configure_cpu_memory(
    cpu: &mut LoongArchCpu,
    ram_ptr: *const u8,
    ram_base: u64,
    ram_size: u64,
    address_space_ptr: u64,
) {
    cpu.set_guest_base(
        (ram_ptr as usize).wrapping_sub(ram_base as usize) as u64
    );
    cpu.set_ram_base(ram_base);
    cpu.set_ram_end(ram_base + ram_size);
    cpu.set_address_space_ptr(address_space_ptr);
}

fn loongarch_tb_flags(cpu: &LoongArchCpu) -> u32 {
    let crmd = cpu.crmd();
    let plv = (crmd & CRMD_PLV_MASK) as u32 & LOONGARCH_TB_FLAG_PLV_MASK;
    let da = if crmd & CRMD_DA != 0 {
        LOONGARCH_TB_FLAG_DA
    } else {
        0
    };
    let pg = if crmd & CRMD_PG != 0 {
        LOONGARCH_TB_FLAG_PG
    } else {
        0
    };
    let fpe = if cpu.euen() & EUEN_FPE != 0 {
        LOONGARCH_TB_FLAG_FPE
    } else {
        0
    };
    plv | da | pg | fpe
}

fn loongarch_timer_counting(cpu: &LoongArchCpu) -> bool {
    cpu.csr_read(CSR_TCFG) & 1 != 0 && cpu.tval() != 0
}

fn loongarch_gen_code(
    cpu: &mut LoongArchCpu,
    ir: &mut Context,
    pc: u64,
    max_insns: u32,
) -> u32 {
    machina_util::trace::trace_tb(pc, loongarch_tb_flags(cpu));
    let phys_pc = match cpu.translate_address_or_exception(
        pc,
        mmu::AccessType::Fetch,
        pc,
    ) {
        Ok(pa) => pa,
        Err(vec) => {
            cpu.set_pc(vec);
            cpu.set_translation_fault_pending();
            return 0;
        }
    };
    cpu.set_last_phys_pc(phys_pc);
    let guest_base = (cpu.guest_base_val() as usize)
        .wrapping_add(phys_pc as usize)
        .wrapping_sub(pc as usize) as *const u8;
    let cfg = LoongArchCfg::default();
    let mut ctx = LoongArchDisasContext::new(pc, guest_base, cfg);
    ctx.base.max_insns = max_insns.min(LOONGARCH_MAX_TB_INSNS);

    if ir.nb_globals() == 0 {
        LoongArchTranslator::init_disas_context(&mut ctx, ir);
    } else {
        ctx.bind_existing_globals(ir);
    }
    LoongArchTranslator::tb_start(&mut ctx, ir);

    loop {
        LoongArchTranslator::insn_start(&mut ctx, ir);
        LoongArchTranslator::translate_insn(&mut ctx, ir);
        if ctx.base.is_jmp != DisasJumpType::Next {
            break;
        }
        if (ctx.base.pc_next & TARGET_PAGE_MASK) != (pc & TARGET_PAGE_MASK) {
            ctx.base.is_jmp = DisasJumpType::TooMany;
            break;
        }
        if ctx.base.num_insns >= ctx.base.max_insns {
            ctx.base.is_jmp = DisasJumpType::TooMany;
            break;
        }
    }

    LoongArchTranslator::tb_stop(&mut ctx, ir);
    ctx.base.num_insns * 4
}

fn loongarch_handle_interrupt(cpu: &mut LoongArchCpu) {
    let Some(irq) = cpu.pending_interrupt_line() else {
        return;
    };
    machina_util::trace::trace_exception(0, cpu.pc());
    let vec = unsafe {
        machina_guest_loongarch::loongarch::trans::helpers
            ::loongarch_helper_raise_exception(
                cpu.env_ptr(),
                0, // ECODE_INT
                0,
            )
    };
    cpu.set_pc(cpu.external_interrupt_vector(irq, vec));
}

fn loongarch_handle_exception(cpu: &mut LoongArchCpu) {
    // LoongArch EXCP_UNDEF -> raise INE (illegal instruction).
    machina_util::trace::trace_exception(0x0D, cpu.pc());
    unsafe {
        let vec = machina_guest_loongarch::loongarch::trans::helpers
            ::loongarch_helper_raise_exception(
                cpu.env_ptr(),
                0x0D, // ECODE_INE
                0,
            );
        cpu.set_pc(vec);
    }
}

fn translate_for_helper(
    cpu: &mut LoongArchCpu,
    gva: u64,
    access: mmu::AccessType,
    size: u32,
) -> Option<u64> {
    let fault_pc = cpu.fault_pc_val();
    let pa = match cpu.translate_address_or_exception(gva, access, fault_pc) {
        Ok(pa) => pa,
        Err(vector) => {
            cpu.set_pc(vector);
            cpu.set_memory_fault_pending(fault_pc);
            return None;
        }
    };

    let addend = helper_fast_tlb_addend(cpu, gva, pa, size);
    cpu.fill_fast_tlb_for_translation(gva, access, pa, addend);
    Some(pa)
}

fn helper_fast_tlb_addend(
    cpu: &LoongArchCpu,
    gva: u64,
    pa: u64,
    size: u32,
) -> usize {
    if let Some(as_) = unsafe { address_space_for(cpu) } {
        if as_.is_mapped(GPA::new(pa), size) {
            return mmu::FAST_TLB_MMIO_ADDEND;
        }
    }
    if pa >= cpu.ram_base_val()
        && pa
            .checked_add(u64::from(size))
            .is_some_and(|end| end <= cpu.ram_end_val())
    {
        return cpu
            .guest_base_val()
            .wrapping_add(pa & mmu::FAST_TLB_PAGE_MASK)
            .wrapping_sub(gva & mmu::FAST_TLB_PAGE_MASK)
            as usize;
    }
    mmu::FAST_TLB_MMIO_ADDEND
}

unsafe fn read_phys_sized(cpu: *const LoongArchCpu, pa: u64, size: u32) -> u64 {
    let cpu_ref = &*cpu;
    let address_space = address_space_for(cpu_ref);
    if let Some(as_) = address_space {
        if as_.is_mapped(GPA::new(pa), size) {
            return as_.read(GPA::new(pa), size);
        }
    }
    if pa >= cpu_ref.ram_base_val()
        && pa
            .checked_add(u64::from(size))
            .is_some_and(|end| end <= cpu_ref.ram_end_val())
    {
        let ptr = (cpu_ref.guest_base_val() + pa) as *const u8;
        match size {
            1 => *ptr as u64,
            2 => (ptr as *const u16).read_unaligned() as u64,
            4 => (ptr as *const u32).read_unaligned() as u64,
            8 => (ptr as *const u64).read_unaligned(),
            _ => 0,
        }
    } else if let Some(as_) = address_space {
        as_.read(GPA::new(pa), size)
    } else {
        0
    }
}

unsafe fn read_phys_bytes(cpu: *const LoongArchCpu, pa: u64, size: u32) -> u64 {
    let mut val = 0_u64;
    for i in 0..size {
        val |= (read_phys_sized(cpu, pa.wrapping_add(u64::from(i)), 1) & 0xff)
            << (i * 8);
    }
    val
}

unsafe fn write_phys_sized(
    cpu: *mut LoongArchCpu,
    pa: u64,
    val: u64,
    size: u32,
) {
    let cpu_ref = &*cpu;
    let address_space = address_space_for(cpu_ref);
    if let Some(as_) = address_space {
        if as_.is_mapped(GPA::new(pa), size) {
            as_.write(GPA::new(pa), size, val);
            return;
        }
    }
    if pa >= cpu_ref.ram_base_val()
        && pa
            .checked_add(u64::from(size))
            .is_some_and(|end| end <= cpu_ref.ram_end_val())
    {
        let ptr = (cpu_ref.guest_base_val() + pa) as *mut u8;
        match size {
            1 => *ptr = val as u8,
            2 => (ptr as *mut u16).write_unaligned(val as u16),
            4 => (ptr as *mut u32).write_unaligned(val as u32),
            8 => (ptr as *mut u64).write_unaligned(val),
            _ => {}
        }
    } else if let Some(as_) = address_space {
        as_.write(GPA::new(pa), size, val);
    }
}

unsafe fn address_space_for(cpu_ref: &LoongArchCpu) -> Option<&AddressSpace> {
    let ptr = cpu_ref.address_space_ptr();
    if ptr == 0 {
        None
    } else {
        Some(&*(ptr as *const AddressSpace))
    }
}

unsafe fn write_phys_bytes(
    cpu: *mut LoongArchCpu,
    pa: u64,
    val: u64,
    size: u32,
) {
    for i in 0..size {
        write_phys_sized(
            cpu,
            pa.wrapping_add(u64::from(i)),
            (val >> (i * 8)) & 0xff,
            1,
        );
    }
}

fn split_page_access(gva: u64, size: u32) -> Option<(u32, u64, u32)> {
    if size == 0 {
        return None;
    }
    let offset = gva & TARGET_PAGE_OFFSET_MASK;
    if offset + u64::from(size) <= TARGET_PAGE_SIZE {
        return None;
    }
    let first_len = (TARGET_PAGE_SIZE - offset) as u32;
    let second_gva = gva.wrapping_add(u64::from(first_len));
    Some((first_len, second_gva, size - first_len))
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu` for the duration of the call.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_mem_read(
    env: *mut u8,
    gva: u64,
    size: u32,
) -> u64 {
    let cpu = &mut *(env as *mut LoongArchCpu);
    let Some(pa) = translate_for_helper(cpu, gva, mmu::AccessType::Load, size)
    else {
        return 0;
    };
    let Some((first_len, second_gva, second_len)) =
        split_page_access(gva, size)
    else {
        return read_phys_sized(cpu, pa, size);
    };
    let Some(second_pa) = translate_for_helper(
        cpu,
        second_gva,
        mmu::AccessType::Load,
        second_len,
    ) else {
        return 0;
    };

    read_phys_bytes(cpu, pa, first_len)
        | (read_phys_bytes(cpu, second_pa, second_len) << (first_len * 8))
}

/// # Safety
/// `env` must point to a valid `LoongArchCpu` for the duration of the call.
#[no_mangle]
pub unsafe extern "sysv64" fn loongarch_mem_write(
    env: *mut u8,
    gva: u64,
    val: u64,
    size: u32,
) {
    let cpu = &mut *(env as *mut LoongArchCpu);
    let Some(pa) = translate_for_helper(cpu, gva, mmu::AccessType::Store, size)
    else {
        return;
    };
    let Some((first_len, second_gva, second_len)) =
        split_page_access(gva, size)
    else {
        write_phys_sized(cpu, pa, val, size);
        return;
    };
    let Some(second_pa) = translate_for_helper(
        cpu,
        second_gva,
        mmu::AccessType::Store,
        second_len,
    ) else {
        return;
    };

    write_phys_bytes(cpu, pa, val, first_len);
    write_phys_bytes(cpu, second_pa, val >> (first_len * 8), second_len);
}

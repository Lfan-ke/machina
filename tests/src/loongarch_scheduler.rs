use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use machina_accel::exec::{ExecEnv, ExitReason};
use machina_accel::GuestCpu;
use machina_accel::X86_64CodeGen;
use machina_core::address::GPA;
use machina_core::machine::{Machine, MachineOpts};
use machina_guest_loongarch::loongarch::cpu::{
    LoongArchCpu, GUEST_BASE_CPU_OFFSET, NEG_ALIGN_CPU_OFFSET,
};
use machina_guest_loongarch::loongarch::csr::{
    CRMD_DA, CRMD_IE, CSR_CRMD, CSR_ECFG, CSR_EENTRY,
};
use machina_guest_loongarch::loongarch::mmu::{AccessType, TlbLookupResult};
use machina_hw_loongarch::interrupt::LOONGARCH_DEVICE_HWI;
use machina_hw_loongarch::virt_machine::{
    LoongArchVirtMachine, VIRT_EIOINTC_BASE, VIRT_IPI_BASE, VIRT_RAM_BASE,
    VIRT_UART_BASE,
};
use machina_system::loongarch_cpu::{
    loongarch_mem_read, loongarch_mem_write, loongarch_soft_mmu_config,
    LoongArchFullSystemCpu,
};
use machina_system::CpuManager;

const IDLE_OP: u32 = 0b00000110010010001;

fn code15_insn(op: u32, code: u32) -> u32 {
    (op << 15) | (code & 0x7FFF)
}

fn default_opts() -> MachineOpts {
    MachineOpts {
        ram_size: 64 * 1024 * 1024,
        cpu_count: 1,
        kernel: None,
        bios: None,
        bios_builtin: false,
        append: None,
        nographic: false,
        drive: None,
        initrd: None,
        dtb: None,
        loaders: Vec::new(),
        netdev: None,
    }
}

fn write_raw_kernel(insns: &[u32]) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    for insn in insns {
        file.write_all(&insn.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
    file
}

fn take_runtime_cpu(
    machine: &mut LoongArchVirtMachine,
    ram_size: u64,
    stop_flag: Arc<AtomicBool>,
) -> LoongArchFullSystemCpu {
    let (cpu_state, interrupts) = machine.take_runtime_cpu_state().unwrap();
    unsafe {
        LoongArchFullSystemCpu::new_with_interrupts(
            cpu_state,
            machine.ram_block().as_ptr(),
            0,
            ram_size,
            machine.address_space() as *const _ as u64,
            stop_flag,
            interrupts,
        )
    }
}

#[test]
fn task45_loongarch_full_system_cpu_sets_address_space_pointer() {
    let code = [code15_insn(IDLE_OP, 0)];
    let cpu = unsafe {
        LoongArchFullSystemCpu::new(
            LoongArchCpu::new(),
            code.as_ptr().cast::<u8>(),
            0,
            4,
            0x1234_5678,
            Arc::new(AtomicBool::new(true)),
        )
    };
    assert_eq!(cpu.cpu.address_space_ptr(), 0x1234_5678);
}

#[test]
fn task45_loongarch_virt_cpu_runs_under_cpu_manager() {
    let kernel = write_raw_kernel(&[code15_insn(IDLE_OP, 0)]);
    let mut opts = default_opts();
    opts.kernel = Some(kernel.path().to_path_buf());

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).unwrap();
    machine.boot().unwrap();

    let mut backend = X86_64CodeGen::new();
    backend.set_guest_base_offset(GUEST_BASE_CPU_OFFSET);
    backend.mmio = Some(loongarch_soft_mmu_config());
    backend.neg_align_off = i32::try_from(NEG_ALIGN_CPU_OFFSET).unwrap();
    let env = ExecEnv::new(backend);
    let shared = env.shared.clone();

    let mut manager = CpuManager::new();
    let stop_flag = manager.running_flag();
    let (cpu_state, interrupts) = machine.take_runtime_cpu_state().unwrap();
    let wake_interrupts = Arc::clone(&interrupts);
    let cpu = unsafe {
        LoongArchFullSystemCpu::new_with_interrupts(
            cpu_state,
            machine.ram_block().as_ptr(),
            0,
            opts.ram_size,
            machine.address_space() as *const _ as u64,
            Arc::clone(&stop_flag),
            interrupts,
        )
    };
    manager.add_loongarch_cpu(cpu);

    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let exit = unsafe { manager.run(&shared) };
        tx.send(exit).unwrap();
    });
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());

    stop_flag.store(false, Ordering::SeqCst);
    wake_interrupts.set_hwi_interrupt_pending(0, true);
    let exit = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(exit, ExitReason::Halted);
    handle.join().unwrap();
}

#[test]
fn task83_runtime_cpu_owns_jit_state_and_receives_async_uart_irq() {
    let kernel = write_raw_kernel(&[code15_insn(IDLE_OP, 0)]);
    let mut opts = default_opts();
    opts.kernel = Some(kernel.path().to_path_buf());

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).unwrap();
    machine.boot().unwrap();

    let board_cpu = machine.cpu();
    let mut runtime_cpu = take_runtime_cpu(
        &mut machine,
        opts.ram_size,
        Arc::new(AtomicBool::new(true)),
    );
    let runtime_env = runtime_cpu.env_ptr();
    let board_env = {
        let mut board_cpu = board_cpu.lock().unwrap();
        std::ptr::from_mut(&mut *board_cpu).cast::<u8>()
    };
    assert_ne!(runtime_env, board_env);
    runtime_cpu.reset_exit_request();
    assert_eq!(runtime_cpu.cpu.neg_align_val(), 0);

    runtime_cpu.cpu.csr_write(CSR_CRMD, CRMD_DA | CRMD_IE);
    runtime_cpu
        .cpu
        .csr_write(CSR_ECFG, 1_u64 << (2 + u32::from(LOONGARCH_DEVICE_HWI)));
    runtime_cpu.cpu.csr_write(CSR_EENTRY, 0x1000);

    machine
        .address_space()
        .write(GPA::new(VIRT_UART_BASE + 1), 1, 1);
    machine.uart().receive(0x5a);

    assert!(runtime_cpu.has_pending_irq());
    runtime_cpu.reset_exit_request();
    assert_eq!(runtime_cpu.cpu.neg_align_val(), -1);
    assert!(runtime_cpu.pending_interrupt());
    runtime_cpu.handle_interrupt();
    assert_eq!(runtime_cpu.cpu.pc(), 0x1000);
}

#[test]
fn task88_runtime_cpu_uses_low_physical_ram_for_direct_map_alias() {
    let opts = default_opts();
    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).unwrap();
    let ram_ptr = machine.ram_block().as_ptr() as usize;

    let mut runtime_cpu = take_runtime_cpu(
        &mut machine,
        opts.ram_size,
        Arc::new(AtomicBool::new(true)),
    );
    runtime_cpu.cpu.csr_write(CSR_CRMD, CRMD_DA);

    assert_eq!(runtime_cpu.cpu.ram_base_val(), 0);
    assert_eq!(runtime_cpu.cpu.ram_end_val(), opts.ram_size);

    let va = VIRT_RAM_BASE + 0x1000;
    match runtime_cpu
        .cpu
        .translate_address_and_cache(va, AccessType::Fetch)
    {
        TlbLookupResult::Hit { pa, .. } => assert_eq!(pa, 0x1000),
        fault => {
            panic!("direct-map alias should translate to low RAM: {fault:?}")
        }
    }

    let addend = runtime_cpu
        .cpu
        .fast_tlb_lookup_addend(va, AccessType::Fetch)
        .expect("direct-map RAM fetch should populate fast TLB");
    assert_eq!(
        addend.wrapping_add(va as usize),
        ram_ptr + 0x1000,
        "fast TLB must target the low physical RAM backing"
    );
}

#[test]
fn task85_runtime_low_mmio_dispatches_before_low_ram_fast_path() {
    let mut opts = default_opts();
    opts.ram_size = 64 * 1024 * 1024;

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).unwrap();

    let mut runtime_cpu = take_runtime_cpu(
        &mut machine,
        opts.ram_size,
        Arc::new(AtomicBool::new(true)),
    );
    runtime_cpu.cpu.csr_write(CSR_CRMD, CRMD_DA);

    let ipi_enable_pa = VIRT_IPI_BASE + 0x004;
    let eiointc_enable_pa = VIRT_EIOINTC_BASE + 0x0200;
    let ram = machine.ram_block().as_ptr();
    unsafe {
        (ram.add(ipi_enable_pa as usize) as *mut u32)
            .write_unaligned(0x1122_3344);
        (ram.add(eiointc_enable_pa as usize) as *mut u32)
            .write_unaligned(0x5566_7788);

        loongarch_mem_write(
            runtime_cpu.env_ptr(),
            ipi_enable_pa,
            0xa5a5_5a5a,
            4,
        );
        loongarch_mem_write(
            runtime_cpu.env_ptr(),
            eiointc_enable_pa,
            0x0000_00f0,
            4,
        );
    }

    assert_eq!(
        unsafe {
            (ram.add(ipi_enable_pa as usize) as *const u32).read_unaligned()
        },
        0x1122_3344,
        "IPI MMIO must not be shadowed by low runtime RAM"
    );
    assert_eq!(
        unsafe {
            (ram.add(eiointc_enable_pa as usize) as *const u32).read_unaligned()
        },
        0x5566_7788,
        "EIOINTC MMIO must not be shadowed by low runtime RAM"
    );
    assert_eq!(machine.ipi().mmio_read_sized(0, 0x004, 4), 0xa5a5_5a5a);
    assert_eq!(machine.eiointc().mmio_read_sized(0, 0x0200, 4), 0xf0);
}

#[test]
fn task96_runtime_low_mmio_read_does_not_populate_ram_fast_tlb() {
    let mut opts = default_opts();
    opts.ram_size = 64 * 1024 * 1024;

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).unwrap();

    let mut runtime_cpu = take_runtime_cpu(
        &mut machine,
        opts.ram_size,
        Arc::new(AtomicBool::new(true)),
    );
    runtime_cpu.cpu.csr_write(CSR_CRMD, CRMD_DA);

    let ipi_enable_pa = VIRT_IPI_BASE + 0x004;
    let ram = machine.ram_block().as_ptr();
    unsafe {
        (ram.add(ipi_enable_pa as usize) as *mut u32)
            .write_unaligned(0x1122_3344);
        let _ = loongarch_mem_read(runtime_cpu.env_ptr(), ipi_enable_pa, 4);
    }

    assert_eq!(
        runtime_cpu
            .cpu
            .fast_tlb_lookup_addend(ipi_enable_pa, AccessType::Load),
        None,
        "low MMIO must keep the JIT on the AddressSpace slow path"
    );
    assert_eq!(
        unsafe {
            (ram.add(ipi_enable_pa as usize) as *const u32).read_unaligned()
        },
        0x1122_3344
    );
}

#[test]
fn task97_cpu_manager_runs_secondary_loongarch_cpus() {
    let mut opts = default_opts();
    opts.cpu_count = 2;

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).unwrap();
    machine.address_space().write(
        GPA::new(VIRT_RAM_BASE),
        4,
        u64::from(code15_insn(IDLE_OP, 0)),
    );

    let mut manager = CpuManager::new();
    let stop_flag = manager.running_flag();
    let cpu_states = machine.take_runtime_cpu_states().unwrap();
    assert_eq!(cpu_states.len(), 2);
    let wake_secondary = Arc::clone(&cpu_states[1].1);

    for (cpu_state, interrupts) in cpu_states {
        let cpu = unsafe {
            LoongArchFullSystemCpu::new_with_interrupts(
                cpu_state,
                machine.ram_block().as_ptr(),
                0,
                opts.ram_size,
                machine.address_space() as *const _ as u64,
                Arc::clone(&stop_flag),
                interrupts,
            )
        };
        manager.add_loongarch_cpu(cpu);
    }

    let mut backend = X86_64CodeGen::new();
    backend.set_guest_base_offset(GUEST_BASE_CPU_OFFSET);
    backend.mmio = Some(loongarch_soft_mmu_config());
    backend.neg_align_off = i32::try_from(NEG_ALIGN_CPU_OFFSET).unwrap();
    let env = ExecEnv::new(backend);
    let shared = env.shared.clone();

    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let exit = unsafe { manager.run(&shared) };
        tx.send(exit).unwrap();
    });
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());

    stop_flag.store(false, Ordering::SeqCst);
    wake_secondary.set_ipi_interrupt_pending(true);
    let exit = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(exit, ExitReason::Halted);
    handle.join().unwrap();
}

#[test]
fn task97_threaded_loongarch_buffer_full_continues_until_stop() {
    let mut opts = default_opts();
    opts.cpu_count = 2;

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).unwrap();
    machine.address_space().write(
        GPA::new(VIRT_RAM_BASE),
        4,
        u64::from(code15_insn(IDLE_OP, 0)),
    );

    let mut manager = CpuManager::new();
    let stop_flag = manager.running_flag();
    for (cpu_state, interrupts) in machine.take_runtime_cpu_states().unwrap() {
        let cpu = unsafe {
            LoongArchFullSystemCpu::new_with_interrupts(
                cpu_state,
                machine.ram_block().as_ptr(),
                0,
                opts.ram_size,
                machine.address_space() as *const _ as u64,
                Arc::clone(&stop_flag),
                interrupts,
            )
        };
        manager.add_loongarch_cpu(cpu);
    }

    let mut backend = X86_64CodeGen::new();
    backend.set_guest_base_offset(GUEST_BASE_CPU_OFFSET);
    backend.mmio = Some(loongarch_soft_mmu_config());
    backend.neg_align_off = i32::try_from(NEG_ALIGN_CPU_OFFSET).unwrap();
    let env = ExecEnv::with_code_buffer_size(backend, 4096);
    let shared = env.shared.clone();

    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let exit = unsafe { manager.run(&shared) };
        tx.send(exit).unwrap();
    });

    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
    stop_flag.store(false, Ordering::SeqCst);
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        ExitReason::Halted
    );
    handle.join().unwrap();
}

#[test]
fn task84_loongarch_idle_waits_for_async_irq_instead_of_exiting_vm() {
    let kernel = write_raw_kernel(&[code15_insn(IDLE_OP, 0)]);
    let mut opts = default_opts();
    opts.kernel = Some(kernel.path().to_path_buf());

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).unwrap();
    machine.boot().unwrap();

    let mut backend = X86_64CodeGen::new();
    backend.set_guest_base_offset(GUEST_BASE_CPU_OFFSET);
    backend.mmio = Some(loongarch_soft_mmu_config());
    backend.neg_align_off = i32::try_from(NEG_ALIGN_CPU_OFFSET).unwrap();
    let env = ExecEnv::new(backend);
    let shared = env.shared.clone();

    let mut manager = CpuManager::new();
    let stop_flag = manager.running_flag();
    let (cpu_state, interrupts) = machine.take_runtime_cpu_state().unwrap();
    let wake_interrupts = Arc::clone(&interrupts);
    let cpu = unsafe {
        LoongArchFullSystemCpu::new_with_interrupts(
            cpu_state,
            machine.ram_block().as_ptr(),
            0,
            opts.ram_size,
            machine.address_space() as *const _ as u64,
            Arc::clone(&stop_flag),
            interrupts,
        )
    };
    manager.add_loongarch_cpu(cpu);

    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let exit = unsafe { manager.run(&shared) };
        tx.send(exit).unwrap();
    });

    assert!(
        rx.recv_timeout(Duration::from_millis(100)).is_err(),
        "LoongArch idle returned from CpuManager instead of waiting"
    );
    stop_flag.store(false, Ordering::SeqCst);
    wake_interrupts.set_hwi_interrupt_pending(0, true);
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(2)).unwrap(),
        ExitReason::Halted
    );
    handle.join().unwrap();
}

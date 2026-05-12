use std::io::Write;
use std::sync::{Arc, Mutex};

use machina_core::address::GPA;
use machina_core::machine::{LoaderSpec, Machine, MachineOpts, NetdevOpts};
use machina_guest_loongarch::loongarch::csr::{
    CRMD_DA, CRMD_IE, CSR_CPUID, CSR_CRMD, CSR_ECFG,
};
use machina_hw_core::bus::SysBusMapping;
use machina_hw_core::chardev::{ByteCb, CharFrontend, Chardev};
use machina_hw_firmware::keys;
use machina_hw_loongarch::interrupt::LOONGARCH_DEVICE_HWI;
use machina_hw_loongarch::virt_machine::{
    LoongArchVirtMachine, VIRT_CPU_COUNT_MAX, VIRT_EIOINTC_BASE,
    VIRT_EIOINTC_SIZE, VIRT_FLASH0_BASE, VIRT_FLASH0_SIZE, VIRT_FLASH1_BASE,
    VIRT_FLASH1_SIZE, VIRT_FWCFG_BASE, VIRT_FWCFG_SIZE, VIRT_IPI_BASE,
    VIRT_IPI_SIZE, VIRT_LEGACY_IO_BASE, VIRT_LEGACY_IO_SIZE,
    VIRT_LEGACY_IPI_BASE, VIRT_LEGACY_IPI_SIZE, VIRT_LEGACY_IPI_STRIDE,
    VIRT_PCH_MSI_BASE, VIRT_PCH_MSI_SIZE, VIRT_PCH_PIC_BASE, VIRT_PCH_PIC_SIZE,
    VIRT_PCI_CFG_BASE, VIRT_PCI_CFG_SIZE, VIRT_PCI_HT_CFG_BASE, VIRT_RAM_BASE,
    VIRT_RTC_BASE, VIRT_RTC_SIZE, VIRT_UART1_BASE, VIRT_UART1_SIZE,
    VIRT_UART_BASE, VIRT_UART_SIZE, VIRT_VIRTIO_BASE, VIRT_VIRTIO_SIZE,
};

const LOONGARCH_RTC_PCH_IRQ_UNDER_TEST: u32 = 6;
const UART_LSR_OFFSET: u64 = 5;
const UART_LSR_THRE: u64 = 1 << 5;

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

fn assert_mapping(
    mappings: &[SysBusMapping],
    owner: &str,
    base: u64,
    size: u64,
) {
    assert!(
        mappings.iter().any(|mapping| {
            mapping.owner == owner
                && mapping.base == GPA::new(base)
                && mapping.size == size
        }),
        "missing mapping {owner} @ {base:#x} size {size:#x}"
    );
}

fn enable_device_hwi(machine: &LoongArchVirtMachine) {
    let cpu = machine.cpu();
    let mut cpu = cpu.lock().unwrap();
    cpu.csr_write(CSR_CRMD, CRMD_DA | CRMD_IE);
    cpu.csr_write(CSR_ECFG, 1 << (u32::from(LOONGARCH_DEVICE_HWI) + 2));
}

struct CapturingInputChardev {
    input_cb: Arc<Mutex<Option<ByteCb>>>,
}

impl Chardev for CapturingInputChardev {
    fn read(&mut self) -> Option<u8> {
        None
    }

    fn write(&mut self, _data: u8) {}

    fn can_read(&self) -> bool {
        true
    }

    fn start_input(&mut self, cb: ByteCb) {
        *self.input_cb.lock().unwrap() = Some(cb);
    }
}

struct CapturingOutputChardev {
    written: Arc<Mutex<Vec<u8>>>,
}

impl Chardev for CapturingOutputChardev {
    fn read(&mut self) -> Option<u8> {
        None
    }

    fn write(&mut self, data: u8) {
        self.written.lock().unwrap().push(data);
    }

    fn can_read(&self) -> bool {
        false
    }
}

#[test]
fn task42_virt_board_realizes_expected_mmio_map() {
    let mut machine = LoongArchVirtMachine::new();
    let opts = default_opts();
    machine.init(&opts).expect("init loongarch ref");

    assert_eq!(machine.name(), "loongarch64-ref");
    assert_eq!(machine.cpu_count(), 1);
    assert_eq!(machine.ram_size(), opts.ram_size);
    assert_eq!(VIRT_PCH_PIC_SIZE, 0x400);

    let mappings = machine.sysbus().mappings();
    assert_mapping(&mappings, "uart0", VIRT_UART_BASE, VIRT_UART_SIZE);
    assert_mapping(&mappings, "ipi0", VIRT_IPI_BASE, VIRT_IPI_SIZE);
    assert_mapping(&mappings, "eiointc0", VIRT_EIOINTC_BASE, VIRT_EIOINTC_SIZE);
    assert_mapping(&mappings, "pch-pic0", VIRT_PCH_PIC_BASE, VIRT_PCH_PIC_SIZE);
    assert_mapping(&mappings, "pch-msi0", VIRT_PCH_MSI_BASE, VIRT_PCH_MSI_SIZE);
    assert_mapping(&mappings, "ls7a-rtc0", VIRT_RTC_BASE, VIRT_RTC_SIZE);
    assert_mapping(&mappings, "pflash0", VIRT_FLASH0_BASE, VIRT_FLASH0_SIZE);
    assert_mapping(&mappings, "pflash1", VIRT_FLASH1_BASE, VIRT_FLASH1_SIZE);

    assert!(
        machine
            .address_space()
            .is_mapped(GPA::new(VIRT_PCH_PIC_BASE + 0x3e7), 1),
        "PCH-PIC INT_POL must be inside the board mapping"
    );
    assert!(
        !machine
            .address_space()
            .is_mapped(GPA::new(VIRT_PCH_PIC_BASE + VIRT_PCH_PIC_SIZE), 1),
        "PCH-PIC mapping must not extend past the declared 0x400 region"
    );

    machine
        .address_space()
        .write(GPA::new(VIRT_RAM_BASE), 4, 0xfeed_beef);
    assert_eq!(
        machine.address_space().read(GPA::new(VIRT_RAM_BASE), 4),
        0xfeed_beef
    );

    let cpu = machine.cpu();
    let cpu = cpu.lock().unwrap();
    assert_eq!(cpu.ram_base_val(), VIRT_RAM_BASE);
    assert_eq!(cpu.ram_end_val(), VIRT_RAM_BASE + opts.ram_size);
}

#[test]
fn task96_virt_board_maps_hvisor_uart1_console() {
    let mut machine = LoongArchVirtMachine::new();
    machine.init(&default_opts()).expect("init loongarch ref");

    assert_mapping(
        &machine.sysbus().mappings(),
        "uart1",
        VIRT_UART1_BASE,
        VIRT_UART1_SIZE,
    );

    let as_ = machine.address_space();
    assert_eq!(
        as_.read(GPA::new(VIRT_UART1_BASE + UART_LSR_OFFSET), 1)
            & UART_LSR_THRE,
        UART_LSR_THRE,
        "hvisor uart1_putchar waits for LSR.THRE before writing"
    );
    as_.write(GPA::new(VIRT_UART1_BASE), 1, u64::from(b'H'));
    assert_eq!(
        as_.read(GPA::new(VIRT_UART1_BASE + UART_LSR_OFFSET), 1)
            & UART_LSR_THRE,
        UART_LSR_THRE
    );
}

#[test]
fn task96_virt_board_routes_hvisor_uart1_output_to_chardev() {
    let written = Arc::new(Mutex::new(Vec::new()));
    let frontend = CharFrontend::new(Box::new(CapturingOutputChardev {
        written: Arc::clone(&written),
    }));

    let mut machine = LoongArchVirtMachine::new();
    machine.set_uart1_chardev(frontend).unwrap();
    machine.init(&default_opts()).expect("init loongarch ref");

    machine.address_space().write(
        GPA::new(VIRT_UART1_BASE),
        1,
        u64::from(b'H'),
    );

    assert_eq!(&*written.lock().unwrap(), b"H");
}

#[test]
fn task97_virt_board_maps_hvisor_legacy_ipi_for_smp_boot() {
    let mut opts = default_opts();
    opts.cpu_count = 4;

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).expect("init loongarch smp ref");

    assert_eq!(machine.cpu_count(), 4);
    for cpu_id in 0..opts.cpu_count {
        let cpu = machine.cpu_at(cpu_id as usize);
        assert_eq!(cpu.lock().unwrap().csr_read(CSR_CPUID), u64::from(cpu_id));
        let ipi_base =
            VIRT_LEGACY_IPI_BASE + u64::from(cpu_id) * VIRT_LEGACY_IPI_STRIDE;
        assert!(
            machine.address_space().is_mapped(GPA::new(ipi_base), 1)
                && machine.address_space().is_mapped(
                    GPA::new(ipi_base + VIRT_LEGACY_IPI_SIZE - 1),
                    1,
                ),
            "missing legacy IPI MMIO for CPU {cpu_id}",
        );
    }

    let entry = 0x1f000_0000;
    let cpu1_base = VIRT_LEGACY_IPI_BASE + VIRT_LEGACY_IPI_STRIDE;
    machine
        .address_space()
        .write(GPA::new(cpu1_base + 0x20), 8, entry);
    machine
        .address_space()
        .write(GPA::new(cpu1_base + 0x08), 4, 1);

    assert_eq!(machine.ipi().mmio_read_sized(1, 0x20, 8), entry);
    assert_eq!(machine.ipi().mmio_read_sized(1, 0, 4) & 1, 1);
}

#[test]
fn task97_virt_board_rejects_too_many_cpus() {
    let mut opts = default_opts();
    opts.cpu_count = VIRT_CPU_COUNT_MAX + 1;

    let mut machine = LoongArchVirtMachine::new();
    let err = machine
        .init(&opts)
        .expect_err("loongarch64-ref must reject too many CPUs");
    let msg = err.to_string();
    assert!(
        msg.contains(&format!(
            "loongarch64-ref supports at most {VIRT_CPU_COUNT_MAX} CPUs"
        )),
        "missing CPU-count limit error: {msg}"
    );
}

#[test]
fn task98_virt_board_reports_empty_loongarch_pci_config_space() {
    let mut machine = LoongArchVirtMachine::new();
    machine.init(&default_opts()).expect("init loongarch ref");

    for base in [VIRT_PCI_CFG_BASE, VIRT_PCI_HT_CFG_BASE] {
        assert!(machine.address_space().is_mapped(GPA::new(base), 4));
        assert!(machine
            .address_space()
            .is_mapped(GPA::new(base + VIRT_PCI_CFG_SIZE - 4), 4));
        assert_eq!(machine.address_space().read(GPA::new(base), 1), 0xff);
        assert_eq!(machine.address_space().read(GPA::new(base), 2), 0xffff);
        assert_eq!(
            machine.address_space().read(GPA::new(base), 4),
            0xffff_ffff
        );
        assert_eq!(
            machine.address_space().read(GPA::new(base + 0x800), 8),
            0xffff_ffff_ffff_ffff,
        );

        machine.address_space().write(GPA::new(base), 4, 0);
        assert_eq!(
            machine.address_space().read(GPA::new(base), 4),
            0xffff_ffff
        );
    }
}

#[test]
fn task98_virt_board_reports_empty_legacy_io_space() {
    let mut machine = LoongArchVirtMachine::new();
    machine.init(&default_opts()).expect("init loongarch ref");

    let as_ = machine.address_space();
    assert!(as_.is_mapped(GPA::new(VIRT_LEGACY_IO_BASE), 1));
    assert!(
        as_.is_mapped(
            GPA::new(VIRT_LEGACY_IO_BASE + VIRT_LEGACY_IO_SIZE - 1),
            1,
        )
    );
    assert_eq!(as_.read(GPA::new(VIRT_LEGACY_IO_BASE + 0x64), 1), 0xff);
    as_.write(GPA::new(VIRT_LEGACY_IO_BASE + 0x64), 1, 0);
    assert_eq!(as_.read(GPA::new(VIRT_LEGACY_IO_BASE + 0x64), 1), 0xff);
}

#[test]
fn task99_virt_board_applies_raw_loader_to_low_physical_ram() {
    let mut kernel = tempfile::NamedTempFile::new().unwrap();
    kernel.write_all(&[0u8; 4]).unwrap();
    kernel.flush().unwrap();
    let mut payload = tempfile::NamedTempFile::new().unwrap();
    payload.write_all(&[0xaa, 0xbb, 0xcc, 0xdd, 0xee]).unwrap();
    payload.flush().unwrap();

    let mut opts = default_opts();
    opts.kernel = Some(kernel.path().to_path_buf());
    opts.loaders.push(LoaderSpec {
        file: payload.path().to_path_buf(),
        addr: 0x0100_0000,
        force_raw: true,
    });

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).expect("init loongarch ref");
    machine.boot().expect("boot loongarch ref");

    assert_eq!(
        machine
            .address_space()
            .read(GPA::new(VIRT_RAM_BASE + 0x0100_0000), 4),
        0xddcc_bbaa,
    );
    assert_eq!(
        machine
            .address_space()
            .read(GPA::new(VIRT_RAM_BASE + 0x0100_0004), 1),
        0xee,
    );
}

#[test]
fn task42_virt_board_maps_pch_msi_and_routes_vectors_to_cpu_hwi() {
    let mut machine = LoongArchVirtMachine::new();
    machine.init(&default_opts()).expect("init loongarch ref");

    assert_mapping(
        &machine.sysbus().mappings(),
        "pch-msi0",
        VIRT_PCH_MSI_BASE,
        VIRT_PCH_MSI_SIZE,
    );
    assert_eq!(
        machine.address_space().read(GPA::new(VIRT_PCH_MSI_BASE), 4),
        0
    );

    enable_device_hwi(&machine);
    machine
        .address_space()
        .write(GPA::new(VIRT_PCH_MSI_BASE), 4, 32);

    let expected_line = Some(u32::from(LOONGARCH_DEVICE_HWI) + 2);
    assert_eq!(
        machine.cpu().lock().unwrap().pending_interrupt_line(),
        expected_line
    );
    machine.interrupt_cascade().ack_eiointc(0, 32);
    assert_eq!(machine.cpu().lock().unwrap().pending_interrupt_line(), None);

    machine
        .address_space()
        .write(GPA::new(VIRT_PCH_MSI_BASE + 4), 4, 37);
    assert_eq!(
        machine.cpu().lock().unwrap().pending_interrupt_line(),
        expected_line
    );
}

#[test]
fn task42_virt_board_maps_ls7a_rtc_and_routes_alarm_to_cpu_hwi() {
    let mut machine = LoongArchVirtMachine::new();
    machine.init(&default_opts()).expect("init loongarch ref");

    assert_mapping(
        &machine.sysbus().mappings(),
        "ls7a-rtc0",
        VIRT_RTC_BASE,
        VIRT_RTC_SIZE,
    );

    let as_ = machine.address_space();
    assert_eq!(as_.read(GPA::new(VIRT_RTC_BASE + 0x40), 4), 0);
    as_.write(GPA::new(VIRT_RTC_BASE + 0x40), 4, (1 << 11) | (1 << 8));
    let toy_now = as_.read(GPA::new(VIRT_RTC_BASE + 0x2c), 4);
    assert_ne!(toy_now, 0);

    enable_device_hwi(&machine);
    as_.write(GPA::new(VIRT_RTC_BASE + 0x34), 4, toy_now);
    machine.rtc().tick(0);

    let expected_line = Some(u32::from(LOONGARCH_DEVICE_HWI) + 2);
    assert_eq!(
        machine.cpu().lock().unwrap().pending_interrupt_line(),
        expected_line
    );
    machine
        .interrupt_cascade()
        .ack_eiointc(0, LOONGARCH_RTC_PCH_IRQ_UNDER_TEST);
    assert_eq!(machine.cpu().lock().unwrap().pending_interrupt_line(), None);
}

#[test]
fn task42_virt_board_maps_loongarch_pflash_banks() {
    let mut machine = LoongArchVirtMachine::new();
    machine.init(&default_opts()).expect("init loongarch ref");

    assert_mapping(
        &machine.sysbus().mappings(),
        "pflash0",
        VIRT_FLASH0_BASE,
        VIRT_FLASH0_SIZE,
    );
    assert_mapping(
        &machine.sysbus().mappings(),
        "pflash1",
        VIRT_FLASH1_BASE,
        VIRT_FLASH1_SIZE,
    );
    assert!(
        machine.pflash0().realized(),
        "pflash0 device state must be realized through its own MOM lifecycle"
    );
    assert!(
        machine.pflash1().realized(),
        "pflash1 device state must be realized through its own MOM lifecycle"
    );

    let as_ = machine.address_space();
    as_.write(GPA::new(VIRT_FLASH0_BASE + 0x55), 1, 0x98);
    assert_eq!(
        as_.read(GPA::new(VIRT_FLASH0_BASE + 0x40), 1),
        u64::from(b'Q')
    );
    assert_eq!(
        as_.read(GPA::new(VIRT_FLASH0_BASE + 0x44), 1),
        u64::from(b'R')
    );
    assert_eq!(
        as_.read(GPA::new(VIRT_FLASH0_BASE + 0x48), 1),
        u64::from(b'Y')
    );

    as_.write(GPA::new(VIRT_FLASH1_BASE), 1, 0xff);
    as_.write(GPA::new(VIRT_FLASH1_BASE), 1, 0x40);
    as_.write(GPA::new(VIRT_FLASH1_BASE + 0x20), 1, 0x5a);
    as_.write(GPA::new(VIRT_FLASH1_BASE), 1, 0xff);
    assert_eq!(as_.read(GPA::new(VIRT_FLASH1_BASE + 0x20), 1), 0x5a);
}

#[test]
fn task45_virt_board_maps_fw_cfg_mmio_window() {
    let mut machine = LoongArchVirtMachine::new();
    let opts = default_opts();
    machine.init(&opts).expect("init loongarch ref");
    let as_ = machine.address_space();

    assert_mapping(
        &machine.sysbus().mappings(),
        "fw_cfg0",
        VIRT_FWCFG_BASE,
        VIRT_FWCFG_SIZE,
    );

    as_.write(
        GPA::new(VIRT_FWCFG_BASE + 0x08),
        2,
        u64::from(keys::SIGNATURE),
    );
    assert_eq!(as_.read(GPA::new(VIRT_FWCFG_BASE), 4), 0x5145_4d55);

    as_.write(
        GPA::new(VIRT_FWCFG_BASE + 0x08),
        2,
        u64::from(keys::MAX_CPUS),
    );
    assert_eq!(
        as_.read(GPA::new(VIRT_FWCFG_BASE), 1),
        opts.cpu_count as u64
    );
    assert_eq!(as_.read(GPA::new(VIRT_FWCFG_BASE), 1), 0);

    as_.write(
        GPA::new(VIRT_FWCFG_BASE + 0x08),
        2,
        u64::from(keys::NB_CPUS),
    );
    assert_eq!(
        as_.read(GPA::new(VIRT_FWCFG_BASE), 1),
        opts.cpu_count as u64
    );
    assert_eq!(as_.read(GPA::new(VIRT_FWCFG_BASE), 1), 0);

    as_.write(
        GPA::new(VIRT_FWCFG_BASE + 0x08),
        2,
        u64::from(keys::RAM_SIZE),
    );
    let ram_size = (0..8).fold(0u64, |value, shift| {
        value | (as_.read(GPA::new(VIRT_FWCFG_BASE), 1) << (shift * 8))
    });
    assert_eq!(ram_size, opts.ram_size);
}

#[test]
fn task87_virt_board_rejects_unsupported_virtio_net_options() {
    let mut opts = default_opts();
    opts.netdev = Some(NetdevOpts {
        id: "net0".to_string(),
        ifname: "tap0".to_string(),
        mac: Some("52:54:00:12:34:56".to_string()),
    });

    let mut machine = LoongArchVirtMachine::new();
    let err = machine
        .init(&opts)
        .expect_err("loongarch64-ref must reject unsupported virtio-net");
    let msg = err.to_string();
    assert!(
        msg.contains(
            "loongarch64-ref does not support virtio-net-device/-netdev"
        ),
        "missing virtio-net rejection message: {msg}"
    );
}

#[test]
fn task88_virtio_dma_uses_guest_low_ram_base() {
    let mut image = tempfile::NamedTempFile::new().unwrap();
    image.write_all(&[0u8; 512]).unwrap();

    let mut opts = default_opts();
    opts.drive = Some(image.path().to_path_buf());

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).expect("init loongarch ref with drive");

    let (_ram_ptr, ram_base, ram_size) = machine
        .virtio_mmio()
        .expect("virtio-mmio0")
        .shared_state()
        .lock()
        .unwrap()
        .ram_info();
    assert_eq!(
        ram_base, 0,
        "virtio DMA must use the low physical RAM base visible to Linux"
    );
    assert_eq!(ram_size, opts.ram_size);
}

#[test]
fn task42_virt_board_installs_iocsr_and_uart_cascade() {
    let mut machine = LoongArchVirtMachine::new();
    machine.init(&default_opts()).expect("init loongarch ref");

    assert!(
        machine.iocsr_bus().write(0, 0x14c0, 4, 0x0202_0202),
        "board IOCSR bus must accept EIOINTC aliases"
    );
    assert_eq!(
        machine.cpu().lock().unwrap().iocsr_read(0x14c0, 4),
        0x0202_0202,
        "CPU IOCSR dispatcher must read through the board bus"
    );
    assert_eq!(
        machine.eiointc().mmio_read_sized(0, 0x0c0, 4),
        0x0202_0202,
        "CPU IOCSR dispatcher must route EIOINTC aliases to the board device"
    );

    enable_device_hwi(&machine);
    machine
        .address_space()
        .write(GPA::new(VIRT_UART_BASE + 1), 1, 1);
    machine.uart().receive(0x41);

    let expected_line = Some(u32::from(LOONGARCH_DEVICE_HWI) + 2);
    assert_eq!(
        machine.cpu().lock().unwrap().pending_interrupt_line(),
        expected_line
    );
    assert_eq!(
        machine.address_space().read(GPA::new(VIRT_UART_BASE), 1),
        0x41
    );
    assert_eq!(machine.cpu().lock().unwrap().pending_interrupt_line(), None);
}

#[test]
fn task82_virt_board_chardev_input_reaches_uart_rx_fifo() {
    let input_cb = Arc::new(Mutex::new(None));
    let frontend = CharFrontend::new(Box::new(CapturingInputChardev {
        input_cb: Arc::clone(&input_cb),
    }));

    let mut machine = LoongArchVirtMachine::new();
    machine.set_uart_chardev(frontend).unwrap();
    machine.init(&default_opts()).expect("init loongarch ref");

    enable_device_hwi(&machine);
    machine
        .address_space()
        .write(GPA::new(VIRT_UART_BASE + 1), 1, 1);

    let cb = input_cb
        .lock()
        .unwrap()
        .as_ref()
        .expect("UART realize must start chardev input")
        .clone();
    cb.lock().unwrap()(0x5A);

    let expected_line = Some(u32::from(LOONGARCH_DEVICE_HWI) + 2);
    assert_eq!(
        machine.cpu().lock().unwrap().pending_interrupt_line(),
        expected_line
    );
    assert_eq!(
        machine.address_space().read(GPA::new(VIRT_UART_BASE), 1),
        0x5A
    );
    assert_eq!(machine.cpu().lock().unwrap().pending_interrupt_line(), None);
}

#[test]
fn task42_virt_board_realizes_optional_virtio_cascade() {
    let mut image = tempfile::NamedTempFile::new().unwrap();
    image.write_all(&[0u8; 512]).unwrap();

    let mut opts = default_opts();
    opts.drive = Some(image.path().to_path_buf());

    let mut machine = LoongArchVirtMachine::new();
    machine.init(&opts).expect("init loongarch ref");

    assert_mapping(
        &machine.sysbus().mappings(),
        "virtio-mmio0",
        VIRT_VIRTIO_BASE,
        VIRT_VIRTIO_SIZE,
    );
    assert_eq!(
        machine.address_space().read(GPA::new(VIRT_VIRTIO_BASE), 4),
        0x7472_6976
    );

    enable_device_hwi(&machine);
    machine
        .virtio_mmio()
        .expect("virtio-mmio0")
        .shared_state()
        .lock()
        .unwrap()
        .inject_rx(0, 1);

    let expected_line = Some(u32::from(LOONGARCH_DEVICE_HWI) + 2);
    assert_eq!(
        machine.cpu().lock().unwrap().pending_interrupt_line(),
        expected_line
    );
    assert_eq!(
        machine
            .address_space()
            .read(GPA::new(VIRT_VIRTIO_BASE + 0x060), 4),
        1
    );

    machine
        .address_space()
        .write(GPA::new(VIRT_VIRTIO_BASE + 0x064), 4, 1);

    assert_eq!(
        machine
            .address_space()
            .read(GPA::new(VIRT_VIRTIO_BASE + 0x060), 4),
        0
    );
    assert_eq!(machine.cpu().lock().unwrap().pending_interrupt_line(), None);
}

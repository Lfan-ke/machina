use super::cpu::LoongArchCpu;
use super::csr::*;

pub const MTLB_SIZE: usize = 64;
pub const STLB_SETS: usize = 256;
pub const STLB_WAYS: usize = 8;
pub const STLB_SIZE: usize = STLB_SETS * STLB_WAYS;
pub const TLB_TOTAL_SIZE: usize = STLB_SIZE + MTLB_SIZE;
pub const TARGET_VIRT_MASK: u64 = (1_u64 << 48) - 1;
pub const FAST_TLB_SIZE: usize = 256;
pub const FAST_TLB_PAGE_MASK: u64 = !0xFFF_u64;
pub const FAST_TLB_INVALID_TAG: u64 = u64::MAX;
pub const FAST_TLB_MMIO_ADDEND: usize = usize::MAX;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TlbEntry {
    pub vppn: u64,
    pub page_size: u8,
    pub asid: u16,
    pub gid: u8,
    pub g: bool,
    pub valid: bool,
    pub ppn0: u64,
    pub ppn1: u64,
    pub plv0: u8,
    pub plv1: u8,
    pub mat0: u8,
    pub mat1: u8,
    pub d0: bool,
    pub d1: bool,
    pub v0: bool,
    pub v1: bool,
    pub nr0: bool,
    pub nr1: bool,
    pub nx0: bool,
    pub nx1: bool,
    pub rplv0: bool,
    pub rplv1: bool,
}

pub struct LoongArchMmu {
    pub mtlb: [TlbEntry; MTLB_SIZE],
    pub stlb: [[TlbEntry; STLB_WAYS]; STLB_SETS],
    pub fast_tlb: Box<[FastTlbEntry; FAST_TLB_SIZE]>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastTlbEntry {
    pub addr_read: u64,
    pub addr_write: u64,
    pub addr_code: u64,
    pub addend: usize,
    pub dirty: u8,
    phys_page: u64,
}

pub mod fast_tlb_offsets {
    pub const ADDR_READ: usize = 0;
    pub const ADDR_WRITE: usize = 8;
    pub const ADDR_CODE: usize = 16;
    pub const ADDEND: usize = 24;
    pub const DIRTY: usize = 32;
    pub const ENTRY_SIZE: usize = core::mem::size_of::<super::FastTlbEntry>();
}

impl Default for FastTlbEntry {
    fn default() -> Self {
        Self {
            addr_read: FAST_TLB_INVALID_TAG,
            addr_write: FAST_TLB_INVALID_TAG,
            addr_code: FAST_TLB_INVALID_TAG,
            addend: 0,
            dirty: 0,
            phys_page: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlbSlot {
    Stlb { set: usize, way: usize },
    Mtlb { index: usize },
}

pub fn stlb_set_index(va: u64, page_size: u8) -> Option<usize> {
    if page_size == 0 || page_size >= 63 {
        return None;
    }
    Some(
        (((va & TARGET_VIRT_MASK) >> (u32::from(page_size) + 1)) as usize)
            & (STLB_SETS - 1),
    )
}

pub fn stlb_flat_index(set: usize, way: usize) -> Option<usize> {
    if set < STLB_SETS && way < STLB_WAYS {
        Some(way * STLB_SETS + set)
    } else {
        None
    }
}

pub fn mtlb_flat_index(index: usize) -> Option<usize> {
    if index < MTLB_SIZE {
        Some(STLB_SIZE + index)
    } else {
        None
    }
}

pub fn decode_tlb_index(index: usize) -> Option<TlbSlot> {
    if index < STLB_SIZE {
        Some(TlbSlot::Stlb {
            set: index % STLB_SETS,
            way: index / STLB_SETS,
        })
    } else if index < TLB_TOTAL_SIZE {
        Some(TlbSlot::Mtlb {
            index: index - STLB_SIZE,
        })
    } else {
        None
    }
}

pub fn fast_tlb_index(va: u64) -> usize {
    let vpn = va >> 12;
    let h = vpn ^ (vpn >> 8);
    (h as usize) & (FAST_TLB_SIZE - 1)
}

impl LoongArchMmu {
    pub fn new() -> Self {
        const EMPTY: TlbEntry = TlbEntry {
            vppn: 0,
            page_size: 0,
            asid: 0,
            gid: 0,
            g: false,
            valid: false,
            ppn0: 0,
            ppn1: 0,
            plv0: 0,
            plv1: 0,
            mat0: 0,
            mat1: 0,
            d0: false,
            d1: false,
            v0: false,
            v1: false,
            nr0: false,
            nr1: false,
            nx0: false,
            nx1: false,
            rplv0: false,
            rplv1: false,
        };
        Self {
            mtlb: [EMPTY; MTLB_SIZE],
            stlb: [[EMPTY; STLB_WAYS]; STLB_SETS],
            fast_tlb: Box::new([FastTlbEntry::default(); FAST_TLB_SIZE]),
        }
    }
}

impl Default for LoongArchMmu {
    fn default() -> Self {
        Self::new()
    }
}

impl LoongArchMmu {
    pub fn flush_fast_tlb(&mut self) {
        self.fast_tlb.fill(FastTlbEntry::default());
    }

    pub fn fast_tlb_lookup_addend(
        &self,
        va: u64,
        access: AccessType,
    ) -> Option<usize> {
        let entry = self.fast_tlb[fast_tlb_index(va)];
        let tag = va & FAST_TLB_PAGE_MASK;
        let matched = match access {
            AccessType::Load => entry.addr_read == tag,
            AccessType::Store => entry.addr_write == tag,
            AccessType::Fetch => entry.addr_code == tag,
        };
        if matched && entry.addend != FAST_TLB_MMIO_ADDEND {
            Some(entry.addend)
        } else {
            None
        }
    }

    pub fn fill_fast_tlb(
        &mut self,
        va: u64,
        access: AccessType,
        pa: u64,
        addend: usize,
    ) {
        let tag = va & FAST_TLB_PAGE_MASK;
        let entry = &mut self.fast_tlb[fast_tlb_index(va)];
        let same_page_or_empty =
            [entry.addr_read, entry.addr_write, entry.addr_code]
                .into_iter()
                .all(|existing| {
                    existing == FAST_TLB_INVALID_TAG || existing == tag
                });
        if !same_page_or_empty {
            *entry = FastTlbEntry::default();
        }

        let access_tag = if addend == FAST_TLB_MMIO_ADDEND {
            FAST_TLB_INVALID_TAG
        } else {
            tag
        };

        match access {
            AccessType::Load => entry.addr_read = access_tag,
            AccessType::Store => entry.addr_write = access_tag,
            AccessType::Fetch => entry.addr_code = access_tag,
        }
        entry.addend = addend;
        entry.dirty = 0;
        entry.phys_page = pa >> 12;
    }

    pub fn entry(&self, index: usize) -> Option<&TlbEntry> {
        match decode_tlb_index(index)? {
            TlbSlot::Stlb { set, way } => Some(&self.stlb[set][way]),
            TlbSlot::Mtlb { index } => Some(&self.mtlb[index]),
        }
    }

    pub fn entry_mut(&mut self, index: usize) -> Option<&mut TlbEntry> {
        match decode_tlb_index(index)? {
            TlbSlot::Stlb { set, way } => Some(&mut self.stlb[set][way]),
            TlbSlot::Mtlb { index } => Some(&mut self.mtlb[index]),
        }
    }

    pub fn write_entry(&mut self, index: usize, entry: TlbEntry) -> bool {
        if let Some(slot) = self.entry_mut(index) {
            *slot = entry;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlbLookupResult {
    Hit { pa: u64, mat: u8 },
    Miss,
    Invalid,
    Dirty,
    PrivViolation,
    ReadProtect,
    ExecProtect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Load,
    Store,
    Fetch,
}

pub fn dmw_match_with(crmd: u64, dmw: &[u64; 4], va: u64) -> Option<u64> {
    let plv = (crmd & CRMD_PLV_MASK) as u8;
    for &window in dmw {
        if window == 0 {
            continue;
        }
        let plv_bit = 1_u64 << u64::from(plv);
        if window & plv_bit == 0 {
            continue;
        }
        let vseg = (window >> 60) & 0xF;
        let va_vseg = (va >> 60) & 0xF;
        if vseg == va_vseg {
            return Some(va & TARGET_VIRT_MASK);
        }
    }
    None
}

pub fn dmw_match(cpu: &LoongArchCpu, va: u64) -> Option<u64> {
    dmw_match_with(cpu.crmd(), &cpu.dmw, va)
}

pub fn direct_map_address_with(
    crmd: u64,
    dmw: &[u64; 4],
    va: u64,
) -> Option<u64> {
    if crmd & CRMD_DA != 0 && crmd & CRMD_PG == 0 {
        return Some(va & TARGET_VIRT_MASK);
    }
    dmw_match_with(crmd, dmw, va)
}

pub fn direct_map_address(cpu: &LoongArchCpu, va: u64) -> Option<u64> {
    direct_map_address_with(cpu.crmd(), &cpu.dmw, va)
}

impl LoongArchMmu {
    pub fn tlb_lookup(
        &self,
        va: u64,
        asid: u16,
        gid: u8,
        stlb_ps: u8,
        access: AccessType,
        plv: u8,
    ) -> TlbLookupResult {
        if let Some(r) = self.search_mtlb(va, asid, gid, access, plv) {
            return r;
        }
        self.search_stlb(va, asid, gid, stlb_ps, access, plv)
    }

    fn search_mtlb(
        &self,
        va: u64,
        asid: u16,
        gid: u8,
        access: AccessType,
        plv: u8,
    ) -> Option<TlbLookupResult> {
        for entry in &self.mtlb {
            if !entry.valid {
                continue;
            }
            if let Some(r) = check_entry(entry, va, asid, gid, access, plv) {
                return Some(r);
            }
        }
        None
    }

    fn search_stlb(
        &self,
        va: u64,
        asid: u16,
        gid: u8,
        stlb_ps: u8,
        access: AccessType,
        plv: u8,
    ) -> TlbLookupResult {
        if stlb_ps == 0 {
            return TlbLookupResult::Miss;
        }
        let Some(set_idx) = stlb_set_index(va, stlb_ps) else {
            return TlbLookupResult::Miss;
        };
        for entry in &self.stlb[set_idx] {
            if !entry.valid {
                continue;
            }
            if let Some(r) = check_entry(entry, va, asid, gid, access, plv) {
                return r;
            }
        }
        TlbLookupResult::Miss
    }
}

fn check_entry(
    entry: &TlbEntry,
    va: u64,
    asid: u16,
    gid: u8,
    access: AccessType,
    plv: u8,
) -> Option<TlbLookupResult> {
    let ps = entry.page_size;
    if ps >= 63 {
        return None;
    }
    let page_mask = (1u64 << ps) - 1;
    let vppn_mask = !((1u64 << (ps + 1)) - 1);
    if (va & vppn_mask) != (entry.vppn << 13 & vppn_mask) {
        return None;
    }
    if !entry.g && entry.asid != asid {
        return None;
    }
    if entry.gid != gid {
        return None;
    }
    let odd = (va >> ps) & 1 != 0;
    let (ppn, v, d, plv_ok, nr, nx, rplv) = if odd {
        (
            entry.ppn1,
            entry.v1,
            entry.d1,
            entry.plv1,
            entry.nr1,
            entry.nx1,
            entry.rplv1,
        )
    } else {
        (
            entry.ppn0,
            entry.v0,
            entry.d0,
            entry.plv0,
            entry.nr0,
            entry.nx0,
            entry.rplv0,
        )
    };
    if !v {
        return Some(TlbLookupResult::Invalid);
    }
    if access == AccessType::Fetch && nx {
        return Some(TlbLookupResult::ExecProtect);
    }
    if access == AccessType::Load && nr {
        return Some(TlbLookupResult::ReadProtect);
    }
    if (!rplv && plv > plv_ok) || (rplv && plv != plv_ok) {
        return Some(TlbLookupResult::PrivViolation);
    }
    if access == AccessType::Store && !d {
        return Some(TlbLookupResult::Dirty);
    }
    let ppn = if ps > 12 {
        ppn & !((1_u64 << (u32::from(ps) - 12)) - 1)
    } else {
        ppn
    };
    let pa = (ppn << 12) | (va & page_mask);
    let mat = if odd { entry.mat1 } else { entry.mat0 };
    Some(TlbLookupResult::Hit { pa, mat })
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy)]
pub struct LoongArchCfg {
    pub has_fpu: bool,
    pub has_lsx: bool,
    pub has_lasx: bool,
    pub has_lbt: bool,
    pub has_lvz: bool,
}

impl LoongArchCfg {
    #[must_use]
    pub const fn cpucfg2(&self) -> u64 {
        let mut val = 0x0060_C00C;
        if self.has_fpu {
            val |= 0x3;
        }
        if self.has_lsx {
            val |= 1 << 6;
        }
        if self.has_lasx {
            val |= 1 << 7;
        }
        if self.has_lbt {
            val |= (1 << 18) | (1 << 19) | (1 << 20);
        }
        if self.has_lvz {
            val |= (1 << 10) | (1 << 11);
        }
        val
    }
}

impl Default for LoongArchCfg {
    fn default() -> Self {
        Self {
            has_fpu: true,
            has_lsx: false,
            has_lasx: false,
            has_lbt: false,
            has_lvz: true,
        }
    }
}

use std::{
    alloc::{alloc, alloc_zeroed, dealloc, Layout},
    mem,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

#[derive(Debug)]
pub struct OwnedBuffer {
    ptr: NonNull<f64>,
    len: usize,
    layout: Option<Layout>,
}

impl OwnedBuffer {
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                layout: None,
            };
        }

        let bytes = len
            .checked_mul(mem::size_of::<f64>())
            .expect("buffer size overflow");
        let layout = Layout::from_size_align(bytes, 32).expect("invalid layout");

        // let ptr = unsafe { alloc_zeroed(layout) as *mut f64 };
        let ptr = unsafe { alloc(layout) as *mut f64 };
        let ptr = NonNull::new(ptr).expect("allocation failed");

        Self {
            ptr,
            len,
            layout: Some(layout),
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const f64 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.as_slice().iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f64> {
        self.as_mut_slice().iter_mut()
    }

    #[inline]
    pub fn fill(&mut self, value: f64) {
        self.as_mut_slice().fill(value);
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl Deref for OwnedBuffer {
    type Target = [f64];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for OwnedBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl AsRef<[f64]> for OwnedBuffer {
    #[inline]
    fn as_ref(&self) -> &[f64] {
        self.as_slice()
    }
}

impl AsMut<[f64]> for OwnedBuffer {
    #[inline]
    fn as_mut(&mut self) -> &mut [f64] {
        self.as_mut_slice()
    }
}

#[derive(Clone, Copy)]
pub struct VecView<'a> {
    data: &'a [f64],
}

pub struct VecViewMut<'a> {
    data: &'a mut [f64],
}

impl<'a> VecView<'a> {
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.data
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> VecViewMut<'a> {
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data
    }
}

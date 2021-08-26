//! Utilities used in bagheer

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io;
use std::io::Error;

use crate::errors;

///Generic struct to hold a value together with its index.
pub struct IndexedTuple<T: Ord> {
    index: usize,
    value: T,
}

impl<T: Ord> IndexedTuple<T> {
    pub fn new(index: usize, value: T) -> Self {
        IndexedTuple {
            index,
            value,
        }
    }
    /// Returns the index of the [`Self`] instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::utils::IndexedTuple;
    /// let tup = IndexedTuple::<u8>::new(3usize, 128u8);
    /// assert_eq!(tup.index(), 3usize);
    /// ```
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the value of the [`Self`] instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::utils::IndexedTuple;
    /// let tup = IndexedTuple::<u8>::new(3usize, 128u8);
    /// assert_eq!(tup.value(), &128u8);
    /// ```
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T: Ord> Eq for IndexedTuple<T> {}

impl<T: Ord> PartialEq<Self> for IndexedTuple<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value().eq(other.value())
    }
}

impl<T: Ord> PartialOrd<Self> for IndexedTuple<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value().partial_cmp(other.value())
    }
}

impl<T: Ord> Ord for IndexedTuple<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value().cmp(other.value())
    }
}

#[derive(PartialEq, PartialOrd)]
/// Generic struct to hold all non-NaN floating point numbers.
pub struct NoNaN<T: num_traits::Float>(T);

impl<T: num_traits::Float> NoNaN<T> {
    /// Creates a new instance of [`Self`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::utils::NoNaN;
    /// let v = vec![1.3f64,1.5f64,0.34f64];
    /// let num = NoNaN::<f64>::new(v[2]).unwrap();
    /// assert_ne!(num.value(), v[1]);
    /// assert_eq!(num.value(), v[2]);
    /// ```
    pub fn new(value: T) -> Option<Self> {
        if value.is_nan() {
            None
        } else {
            Some(NoNaN(value))
        }
    }

    /// Returns the non-NaN primitive floating point number inside the [`Self`] instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::utils::NoNaN;
    /// let v = vec![1.3f64,1.5f64,0.34f64];
    /// let num = NoNaN::<f64>::new(v[2]).unwrap();
    /// assert_ne!(num.value(), v[1]);
    /// assert_eq!(num.value(), v[2]);
    /// ```
    pub fn value(&self) -> T {
        self.0
    }
}

impl<T: num_traits::Float> Eq for NoNaN<T> {}

impl<T: num_traits::Float> Ord for NoNaN<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub trait TopK {
    fn top_k(&self, k: usize) -> Result<Vec<usize>, io::Error>;
}




macro_rules! impl_topk_non_float {
    ($($ty:ty),*) =>{
        $(
        impl TopK for Vec<$ty>{
            fn top_k(&self, k: usize) -> Result<Vec<usize>, Error> {
                if k > self.len(){
                    return Err(errors::topk_incorrect_k(k, self.len()));
                }

                let mut bheap = BinaryHeap::<IndexedTuple<$ty>>::with_capacity(self.len());
                for (index, value) in self.iter().enumerate(){
                bheap.push(
                    IndexedTuple::new(index, *value)
                );
                }

                let mut topk_indices = Vec::<usize>::with_capacity(k);
                for _ in 0usize..k{
                    topk_indices.push(
                        bheap.pop().unwrap().index()
                    );
                }
                Ok(topk_indices)
            }
        }
        )*
    }
}


macro_rules! impl_topk_float {
    ($($ty:ty),*) =>{
        $(
        impl TopK for Vec<$ty>{
            fn top_k(&self, k: usize) -> Result<Vec<usize>, Error> {
                if k > self.len(){
                    return Err(errors::topk_incorrect_k(k, self.len()));
                }

                let mut bheap = BinaryHeap::<IndexedTuple<NoNaN<$ty>>>::with_capacity(self.len());
                for (index, value) in self.iter().enumerate(){
                bheap.push(
                    IndexedTuple::new(index, NoNaN::new(*value).unwrap())
                );
                }

                let mut topk_indices = Vec::<usize>::with_capacity(k);
                for _ in 0usize..k{
                    topk_indices.push(
                        bheap.pop().unwrap().index()
                    );
                }
                Ok(topk_indices)
            }
        }
        )*
    }
}

impl_topk_float!(f32, f64);
impl_topk_non_float!(u8, i8, u16, i16, u32, i32, u64, i64,  u128, i128, usize);
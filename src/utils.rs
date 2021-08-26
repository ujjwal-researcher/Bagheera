//! Utilities used in bagheer

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io;

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
//
// pub trait TopK<T> {
//     fn top_k(&self, k: usize) -> Result<Vec<usize>, io::Error> {
//         if k > self.len() {
//             return Err(errors::topk_incorrect_k(k, self.len()));
//         }
//         let bheap = Self::prepare_indexed_heap(&self);
//         let mut topk_indices = Vec::<usize>::with_capacity(k);
//         for _ in 0..k {
//             topk_indices.push(
//                 (*bheap).pop().unwrap().index()
//             )
//         }
//         Ok(topk_indices)
//     }
//
//     fn prepare_indexed_heap(&self) -> &mut BinaryHeap<IndexedTuple<T>> where T : Ord;
// }
//
//
// // impl<T> TopK<T> for Vec<T> where T: num_traits::Unsigned + Copy {
// //     fn prepare_indexed_heap(&self) -> &mut BinaryHeap<IndexedTuple<T>> {
// //         let mut bheap = BinaryHeap::<IndexedTuple<T>>::with_capacity(self.len());
// //         for (index, value) in self.iter().enumerate() {
// //             bheap.push(
// //                 IndexedTuple::new(index, *value)
// //             );
// //         }
// //         &mut bheap
// //     }
// // }
//
// impl<T> TopK<T> for Vec<T> where T : num_traits::Float + Copy{
//     fn prepare_indexed_heap(&self) -> &mut BinaryHeap<IndexedTuple<T>> where  {
//         let mut bheap = BinaryHeap::<IndexedTuple<NoNaN<T>>>::with_capacity(self.len());
//         for (index, value) in self.iter().enumerate() {
//             bheap.push(
//                 IndexedTuple::new(index, NoNaN::new(*value).unwrap())
//             );
//         }
//         &mut bheap
//     }
// }

pub trait TopK<T> {
    fn top_k(&self, k: usize) -> Result<Vec<usize>, io::Error> {
        if k > self.len() {
            return Err(errors::topk_incorrect_k(k, self.len()));
        }
        let bheap = Self::prepare_indexed_heap(&self);
        let mut topk_indices = Vec::<usize>::with_capacity(k);
        for _ in 0..k {
            topk_indices.push(
                (*bheap).pop().unwrap().index()
            )
        }
        Ok(topk_indices)
    }

    fn prepare_indexed_heap(&self) -> &mut BinaryHeap<IndexedTuple<T>> where T : Ord;
}


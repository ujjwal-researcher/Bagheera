//! Utilities used in bagheer

///Generic struct to hold a value together with its index.
pub struct IndexedTuple<T: Ord> {
    index: usize,
    value: T,
}
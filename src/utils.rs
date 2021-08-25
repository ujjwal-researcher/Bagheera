//! Utilities used in bagheer

///Generic struct to hold a value together with its index.
pub struct IndexedTuple<T: Ord> {
    index: usize,
    value: T,
}

impl<T: Ord> IndexedTuple<T>{
    pub fn new(index : usize, value : T) -> Self{
        IndexedTuple{
            index,
            value
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
    pub fn index(&self) -> usize{
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
    pub fn value(&self) -> &T{
        &self.value
    }

}
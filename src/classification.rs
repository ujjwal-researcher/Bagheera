use crate::errors;
use std::collections::HashMap;
use std::io::{Error, ErrorKind};

/// Generic struct representing an image classification dataset.
pub struct ClassificationDataset<
    T1: num_traits::PrimInt + num_traits::Unsigned + num_traits::FromPrimitive,
> {
    num_classes: T1,
    data: HashMap<String, Vec<T1>>,
    is_multilabel: bool,
}

impl<T1: num_traits::PrimInt + num_traits::Unsigned + num_traits::FromPrimitive>
    ClassificationDataset<T1>
{
    /// Returns a new empty instance of [`Self<T1>`].
    ///
    /// Images can be added to the instance using the [`Self::add()`] function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    ///
    /// let cls_db = ClassificationDataset::new(30u8, false);
    /// assert_eq!(cls_db.num_classes(), 30u8);
    /// ```
    pub fn new(num_classes: T1, is_multilabel: bool) -> Self {
        ClassificationDataset {
            num_classes,
            data: HashMap::<String, Vec<T1>>::new(),
            is_multilabel,
        }
    }
    /// Adds a new GT to the [`Self`] instance.
    ///
    /// If `imagename` is already in [`Self`] instance, an [Error] instance is returned.
    /// In case of issues during one-hot conversion, a panic happens.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    ///
    /// let mut cls_db = ClassificationDataset::new(5u16, false);
    /// cls_db.add("hello.jpg", &vec![1u16]);
    /// assert_eq!(cls_db.num_images(), 1usize);
    /// ```
    pub fn add(&mut self, imagename: &str, category_labels: &Vec<T1>) -> Result<(), Error> {
        if self.image_is_present(imagename) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Image {} was already present.", imagename),
            ));
        }
        if !self.is_multilabel && category_labels.len() > 1 {
            return Err(
                Error::new(
                    ErrorKind::InvalidData,
                    "Tried adding multi-label data to a ClassificationDataset instance that is not multilabel.",
                )
            );
        }
        self.data
            .insert(imagename.to_string(), category_labels.to_owned());
        Ok(())
    }
    /// Gets the groundtruth for `imagename` in [`Self`] instance.
    ///
    /// If `imagename` is not in the [`Self`] instance, an [Error]
    /// instance is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    ///
    /// let mut cls_db = ClassificationDataset::new(5u16, false);
    /// cls_db.add("hello.jpg", &vec![1u16]);
    /// assert_eq!(cls_db.get_gt("hello.jpg").unwrap(), &vec![1u16]);
    /// ```
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    ///
    /// let mut cls_db = ClassificationDataset::new(5u16, true);
    /// cls_db.add("hello.jpg", &vec![1u16, 3u16]);
    /// assert_eq!(cls_db.get_gt("hello.jpg").unwrap(), &vec![1u16, 3u16]);
    /// ```
    #[inline]
    pub fn get_gt(&self, imagename: &str) -> Result<&Vec<T1>, Error> {
        if !self.image_is_present(imagename) {
            Err(errors::image_not_present_error(imagename))
        } else {
            Ok(&self.data[imagename])
        }
    }
    /// Returns true if a [`Self`] instance is for multi-label classification.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    /// let cls_db = ClassificationDataset::new(1000u32, false);
    /// assert_eq!(cls_db.is_multilabel(), false);
    /// ```
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    /// let cls_db = ClassificationDataset::new(100u8, true);
    /// assert_eq!(cls_db.is_multilabel(), true);
    /// ```
    #[inline(always)]
    pub fn is_multilabel(&self) -> bool {
        self.is_multilabel
    }
    /// Returns the number of object classes in the [`Self`] instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    ///  let cls_db = ClassificationDataset::<u32>::new(20u32, false);
    ///  assert_eq!(cls_db.num_classes(), 20u32);
    /// ```
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    ///  let cls_db = ClassificationDataset::<u128>::new(2000u128, true);
    ///  assert_eq!(cls_db.num_classes(), 2000u128);
    /// ```
    #[inline(always)]
    pub fn num_classes(&self) -> T1 {
        self.num_classes
    }
    /// Returns the number of images in a [`Self`] instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    /// let mut cls_db = ClassificationDataset::new(100u16, false);
    /// cls_db.add("hello.jpg", &vec![11u16]);
    /// assert_eq!(cls_db.num_images(), 1);
    /// ```
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    /// let cls_db = ClassificationDataset::new(1000u32, false);
    /// assert_eq!(cls_db.num_images(), 0);
    /// ```
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    /// let mut cls_db = ClassificationDataset::new(1000u64, true);
    /// cls_db.add("foo.jpg", &vec![11u64, 3u64, 440u64]);
    /// assert_eq!(cls_db.num_images(), 1);
    /// ```
    #[inline(always)]
    pub fn num_images(&self) -> usize {
        self.data.len()
    }
    /// Returns true if `imagename` is in a [`Self`] instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    /// let mut cls_db = ClassificationDataset::new(100u8, false);
    /// cls_db.add("abc.jpg", &vec![11u8]);
    /// cls_db.add("mango.png", &vec![3u8]);
    /// assert_eq!(cls_db.image_is_present("abc.jpg"), true);
    /// assert_ne!(cls_db.image_is_present("bar.jpg"), true);
    /// ```
    #[inline(always)]
    pub fn image_is_present(&self, imagename: &str) -> bool {
        self.data.contains_key(imagename)
    }
    /// Returns true if the [`Self`] instance is empty i.e has no images.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    /// let cls_db = ClassificationDataset::new(10u8, true);
    /// assert_eq!(cls_db.is_empty(), true);
    /// ```
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    /// let mut cls_db = ClassificationDataset::new(1000u32, false);
    /// cls_db.add("abc.jpg", &vec![74u32]);
    /// assert_eq!(cls_db.is_empty(), false);
    /// ```
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Lists all the images in the [`Self`] instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationDataset;
    /// use std::collections::HashSet;
    /// let mut cls_db = ClassificationDataset::new(100u16, true);
    /// let mut images  = vec!["india.jpg", "italy.png", "iran.jpg", "peru.jpg", "canada.jpg"];
    /// for (index, im) in images.iter().enumerate(){
    ///     cls_db.add(*im, &vec![1u16]);
    /// }
    /// let list_of_images = cls_db.list_images();
    /// let mut lhs = HashSet::<&str>::with_capacity(list_of_images.len());
    /// for item in list_of_images{
    ///     lhs.insert(item);
    /// }
    /// let mut rhs = HashSet::<&str>::with_capacity(images.len());
    /// for item in images{
    ///     rhs.insert(item);
    /// }
    /// assert_eq!(lhs, rhs);
    /// ```
    #[inline(always)]
    pub fn list_images(&self) -> Vec<&str> {
        self.data.keys().map(|x| x.as_str()).collect::<Vec<&str>>()
    }
}

//! Evaluation of image classification
//!
//! Provides abstractions for reading image classification output
//! , reading image classification dataset groundtruth and evaluating
//! single-class and multi-class classification techniques.

use std::collections::HashMap;
use std::io;

use crate::errors;

/// Generic struct to store the image classification output for a number of images.
pub struct ClassificationOutput<T1: num_traits::PrimInt + num_traits::Unsigned + num_traits::FromPrimitive, T2: num_traits::Float + fast_float::FastFloat> {
    num_classes: T1,
    data: HashMap<String, Vec<T2>>,
}

impl<T1: num_traits::PrimInt + num_traits::Unsigned + num_traits::FromPrimitive, T2: num_traits::Float + fast_float::FastFloat> ClassificationOutput<T1, T2> {
    /// Creates a new empty instance of [ClassificaionOutput]
    ///
    /// Items need to be subsequently added to it using [`self::add()`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    /// let cls_out = ClassificationOutput::<u8, f64>::new(20u8);
    /// assert_eq!(cls_out.num_classes(), 20u8);
    /// assert_eq!(cls_out.is_empty(), true);
    /// ```
    ///
    /// ```rust
    /// // The following creates a new empty instance of ClassificationOutput<usize,f32> with     ///
    /// //  1000 classes.
    /// use bagheera::classification::ClassificationOutput;
    /// let mut cls_output = ClassificationOutput::<usize, f32>::new(1000usize);
    /// assert_eq!(cls_output.num_classes(), 1000usize);
    /// assert_eq!(cls_output.is_empty(), true);
    /// ```
    pub fn new(num_classes: T1) -> Self {
        ClassificationOutput {
            num_classes,
            data: HashMap::<String, Vec<T2>>::new(),
        }
    }

    /// Adds a new entry to a [ClassificationOutput] instance.
    ///
    /// This returns an [`io::Error`] instance if the new entry has different number of classes
    /// than that of the [ClassificationOutput] instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    /// let mut cls_out = ClassificationOutput::new(10usize);
    /// let images = vec!["india.jpg", "germany.png", "iran.jpg"];
    /// for img in images{
    ///     let v = vec![1f64; 10];
    ///     cls_out.add(img, v);
    /// }
    /// assert_eq!(cls_out.num_images(), 3usize);
    /// ```
    pub fn add(&mut self, image_name: &str, confidence_vector: Vec<T2>) -> Result<(), io::Error> {
        if confidence_vector.len() == T1::to_usize(&self.num_classes).unwrap() {
            &self.data.insert(image_name.to_string(), confidence_vector);
            Ok(())
        } else {
            Err(errors::image_not_present_error(image_name))
        }
    }
    /// Returns the number of object classes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    ///  let cls_out = ClassificationOutput::<u32, f64>::new(20u32);
    ///  assert_eq!(cls_out.num_classes(), 20u32);
    /// ```
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    ///  let cls_out = ClassificationOutput::<usize, f32>::new(1000usize);
    ///  assert_eq!(cls_out.num_classes(), 1000usize);
    /// ```
    pub fn num_classes(&self) -> T1 {
        self.num_classes
    }

    /// Returns the number of images in a [ClassificationOutput] instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    /// let mut cls_out = ClassificationOutput::<usize,f64>::new(10usize);
    /// let images = vec!["india.jpg", "germany.png", "iran.jpg"];
    /// for img in images{
    ///     let v = vec![1f64; 10];
    ///     cls_out.add(img, v);
    /// }
    /// assert_eq!(cls_out.num_images(), 3usize);
    /// ```
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    /// let mut cls_out = ClassificationOutput::<u8, f32>::new(30u8);
    /// let images = vec!["india.jpg", "germany.png", "iran.jpg", "canada.png", "japan.jpg"];
    /// for img in images{
    ///     let v = vec![1f32; 30];
    ///     cls_out.add(img, v);
    /// }
    /// assert_eq!(cls_out.num_images(), 5usize);
    /// ```
    pub fn num_images(&self) -> usize {
        self.data.len()
    }


    /// Returns true if `image_name` is present in a [ClassificationOutput] instance. False otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    /// let mut cls_out = ClassificationOutput::<u8, f32>::new(30u8);
    /// let images = vec!["india.jpg", "germany.png", "iran.jpg", "canada.png", "japan.jpg"];
    /// for img in images{
    ///     let v = vec![1f32; 30];
    ///     cls_out.add(img, v);
    /// }
    /// assert_eq!(cls_out.image_is_present("australia.jpg"), false);
    /// ```
    pub fn image_is_present(&self, image_name: &str) -> bool {
        self.data.contains_key(image_name)
    }

    /// Returns a sorted vector of image names in a [ClassificationOutput] instance.
    ///
    /// The returned vector contains `&str` slices to the `String` stored in the instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    /// let mut cls_out = ClassificationOutput::<u8, f32>::new(30u8);
    /// let mut images = vec!["india.jpg", "germany.png", "iran.jpg", "canada.png", "japan.jpg"];
    /// images.sort();
    /// for img in &images{
    ///     let v = vec![1f32; 30];
    ///     cls_out.add(img, v);
    /// }
    ///
    /// assert_eq!(cls_out.list_images(), images);
    /// ```
    pub fn list_images(&self) -> Vec<&str> {
        let mut images = Vec::<&str>::with_capacity(self.num_images());
        for (img_name, _) in &self.data {
            images.push(img_name);
        }
        images.sort();
        images
    }


    /// Returns true if a [ClassificationOutput] instance is empty. False otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    /// let cls_out = ClassificationOutput::<u8, f64>::new(20u8);
    /// assert_eq!(cls_out.num_classes(), 20u8);
    /// assert_eq!(cls_out.is_empty(), true);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}



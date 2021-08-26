//! Evaluation of image classification
//!
//! Provides abstractions for reading image classification output
//! , reading image classification dataset groundtruth and evaluating
//! single-class and multi-class classification techniques.

use std::collections::HashMap;
use std::io;

use crate::errors;
use crate::utils;
use crate::utils::TopK;

/// Generic struct to store the image classification output for a number of images.
pub struct ClassificationOutput<T1: num_traits::PrimInt + num_traits::Unsigned + num_traits::FromPrimitive, T2: num_traits::Float + fast_float::FastFloat> {
    num_classes: T1,
    data: HashMap<String, Vec<T2>>,
}

impl<T1: num_traits::PrimInt + num_traits::Unsigned + num_traits::FromPrimitive, T2: num_traits::Float + fast_float::FastFloat> ClassificationOutput<T1, T2> {
    /// Creates a new empty instance of [`Self`]
    ///
    /// Items need to be subsequently added to it using [`Self::add()`]
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

    /// Adds a new entry to a [`Self`] instance.
    ///
    /// This returns an [`io::Error`] instance if the new entry has different number of classes
    /// than that of the [`Self`] instance.
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

    /// Returns the number of images in a [`Self`] instance.
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


    /// Returns true if `image_name` is present in a [`Self`] instance. False otherwise.
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

    /// Returns a sorted vector of image names in a [`Self`] instance.
    ///
    /// The returned vector contains `&str` slices to the `String` stored in the instance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bagheera::classification::ClassificationOutput;
    /// use std::collections::HashSet;
    /// let mut cls_out = ClassificationOutput::<u8, f32>::new(30u8);
    /// let mut images = vec!["india.jpg", "germany.png", "iran.jpg", "canada.png", "japan.jpg"];
    /// images.sort();
    /// for img in &images{
    ///     let v = vec![1f32; 30];
    ///     cls_out.add(img, v);
    /// }
    /// let list_of_images = cls_out.list_images();
    /// let mut lhs = HashSet::<&str>::with_capacity(list_of_images.len());
    /// for item in list_of_images{
    ///     lhs.insert(item);
    /// }
    ///
    /// let mut rhs = HashSet::<&str>::with_capacity(images.len());
    /// for item in images{
    ///     rhs.insert(item);
    /// }
    /// assert_eq!(lhs, rhs);
    /// ```
    pub fn list_images(&self) -> Vec<&str> {
        let mut images = Vec::<&str>::with_capacity(self.num_images());
        for (img_name, _) in &self.data {
            images.push(img_name);
        }
        images
    }


    /// Returns true if a [`Self`] instance is empty. False otherwise.
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

    /// Returns the confidence vector for `imagename` if it exists in the [`Self`] instance.
    /// An [io::Error] instance is returned otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    ///     use bagheera::classification::ClassificationOutput;
    ///     use rand;
    ///     use rand::Rng;
    ///     use float_cmp::approx_eq;
    ///    let mut cls_out = ClassificationOutput::<u32,f32>::new(1000u32);
    ///     let rand_iter = rand::thread_rng().gen_range(0usize..500usize);
    ///     println!("{}",rand_iter);
    ///     let mut test_image = String::new();
    ///     let mut test_vec = Vec::<f32>::new();
    ///    for i in 0usize..500usize{
    ///         let rand_name : String = (0..50).map(|_| rand::random::<u8>() as char).collect();
    ///         let rand_vec = vec![rand::random::<f32>() ; 1000usize];
    ///         if i == rand_iter{
    ///             test_vec = rand_vec.clone();
    ///             test_image = rand_name.clone();
    ///             println!("Test image : {}",test_image);
    ///         }
    ///         cls_out.add(&rand_name, rand_vec);
    ///     }
    ///
    /// for (lhs, rhs) in (*(cls_out.confidence_for_image(&test_image).unwrap())).iter().zip(test_vec.iter()){
    ///     approx_eq!(f32, *lhs, *rhs, ulps=5);
    /// }
    /// ```
    pub fn confidence_for_image(&self, imagename: &str) -> Result<&Vec<T2>, io::Error> {
        if !self.image_is_present(imagename) {
            return Err(errors::image_not_present_error(imagename));
        } else {
            Ok(&self.data[imagename])
        }
    }

    /// Returns indices for the Top-k entries of the confidence vector for an image in
    /// a [`Self`] instance.
    ///
    /// If `imagename` is not in the [`Self`] instance, an [io::Error] instance is returned
    ///
    /// # Examples
    ///
    /// ```rust
    ///     use bagheera::classification::ClassificationOutput;
    ///     use rand;
    ///     use rand::Rng;
    ///     use float_cmp::approx_eq;
    ///    let mut cls_out = ClassificationOutput::<u32,f32>::new(1000u32);
    ///     let rand_iter = rand::thread_rng().gen_range(0usize..500usize);
    ///     let mut test_image = String::new();
    ///     let mut test_vec = Vec::<f32>::new();
    ///    for i in 0usize..500usize{
    ///         let rand_name : String = (0..50).map(|_| rand::random::<u8>() as char).collect();
    ///         let mut rand_vec = Vec::<f32>::new();
    ///         if i == rand_iter{
    ///             test_image = rand_name.clone();
    ///             rand_vec = (0..1000).map(|x| x as f32).collect();
    ///         }
    ///         else {
    ///             rand_vec =  vec![rand::random::<f32>() ; 1000usize];
    ///         }
    ///         cls_out.add(&rand_name, rand_vec);
    ///     }
    ///
    ///     let topk_ind = cls_out.topk_for_image(&test_image,3usize).unwrap();
    ///     assert_eq!(topk_ind, vec![999usize, 998usize, 997usize])
    ///```
    pub fn topk_for_image(&self, imagename: &str, k: usize) -> Result<Vec<usize>, io::Error> where Vec<T2>: utils::TopK {
        if !self.image_is_present(imagename) {
            return Err(errors::image_not_present_error(imagename));
        }
        let topk_indices = (*self.confidence_for_image(imagename).unwrap()).top_k(k).unwrap();

        Ok(topk_indices)
    }
}



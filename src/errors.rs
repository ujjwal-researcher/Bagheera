use std::io;

/// Returns an `io::Error` instance with a custom string.
///
/// This should be used to return an `io::Error` in cases where
/// a specific `image_name` is not found in a HashMap object as a key.
///
/// # Examples
///
/// ```rust
/// use bagheera::classification::ClassificationOutput;
/// use bagheera::errors::image_not_present_error;
/// use std::io;
/// let mut cls_out = ClassificationOutput::<usize,f64>::new(1000usize);
/// let images = vec!["india.jpg", "germany.png", "iran.jpg", "canada.png", "japan.jpg"];
/// for img in &images{
///     let v = vec![1f64; 30];
///     cls_out.add(img, v);
/// }
///
/// let m = match cls_out.image_is_present("peru.jpg"){
///     true => io::Error::new(io::ErrorKind::Other, "Not an error"),
///     _ => image_not_present_error("peru.jpg")
/// };
///
/// assert_eq!(m.kind(), io::ErrorKind::NotFound);
///
/// ```
pub fn image_not_present_error(image_name: &str) -> io::Error {
    io::Error::new(io::ErrorKind::NotFound,
                   format!("The image {} was not found.", image_name),
    )
}

/// Returns an `io::Error` instance with a custom string when K during Top-K analysis of a vector
/// is set higher than the length of the vector itself.
///
/// # Examples
///
/// ```rust
/// ```

pub fn topk_incorrect_k(k: usize, v_length: usize) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, format!("In Top-K analysis of a vector v, K <= v.len().\
    Here K = {} and v.len() = {}.", k, v_length),
    )
}
pub mod classification;
pub mod errors;
pub mod utils;

#[cfg(test)]
mod tests {
    use crate::classification::ClassificationOutput;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn empty_cls_output_u8_f64() {
        let cls_out = ClassificationOutput::<u8, f64>::new(4u8);
        assert_eq!(cls_out.num_classes(), 4u8);
        assert_eq!(cls_out.is_empty(), true);
    }

    #[test]
    fn empty_cls_output_usize_f32() {
        let cls_out = ClassificationOutput::<usize, f32>::new(1000usize);
        assert_eq!(cls_out.num_classes(), 1000usize);
        assert_eq!(cls_out.num_images(), 0usize);
        assert_eq!(cls_out.is_empty(), true);
    }

    #[test]
    fn nonempty_cls_output_u16_f32() {
        let mut cls_out = ClassificationOutput::<u16, f32>::new(30u16);
        let images = vec!["abc.jpg", "cde.jpg", "efg.jpg"];
        for (index, image) in images.iter().enumerate() {
            cls_out.add(*image, vec![index as f32; 30]).unwrap();
        }
        assert_eq!(cls_out.num_images(), 3usize);
    }

    #[test]
    fn test_num_classes() {
        let cls_out = ClassificationOutput::<u128, f64>::new(1000u128);
        assert_eq!(cls_out.num_classes(), 1000u128);
    }

    #[test]
    fn classification_test_image_is_present() {
        let mut cls_out = ClassificationOutput::<u8, f32>::new(30u8);
        let images = vec!["india.jpg", "germany.png", "iran.jpg", "canada.png", "japan.jpg"];
        for img in images {
            let v = vec![1f32; 30];
            cls_out.add(img, v);
        }
        assert_eq!(cls_out.image_is_present("australia.jpg"), false);
    }

    #[test]
    fn classification_test_list_images() {
        let mut cls_out = ClassificationOutput::<u8, f32>::new(30u8);
        let mut images = vec!["india.jpg", "germany.png", "iran.jpg", "canada.png", "japan.jpg"];
        images.sort();
        for img in &images {
            let v = vec![1f32; 30];
            cls_out.add(*img, v);
        }
        assert_eq!(cls_out.list_images(), images);
    }

    #[test]
    fn classification_test_classification_output_is_empty() {
        let cls_out = ClassificationOutput::<u8, f64>::new(20u8);
        assert_eq!(cls_out.is_empty(), true);
    }
}

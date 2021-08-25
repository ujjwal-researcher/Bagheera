pub mod classification;
pub mod errors;

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
            cls_out.add(*image, vec![index as f32; 30]);
        }
        assert_eq!(cls_out.num_images(), 3usize);
    }

    #[test]
    fn test_num_classes() {
        let cls_out = ClassificationOutput::<u128, f64>::new(1000u128);
        assert_eq!(cls_out.num_classes(), 1000u128);
    }
}

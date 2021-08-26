pub mod classification;
pub mod errors;
pub mod utils;

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use rand;
    use rand::Rng;

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
            cls_out.add(img, v).unwrap();
        }
        assert_eq!(cls_out.image_is_present("australia.jpg"), false);
    }

    #[test]
    fn classification_test_list_images() {
        use std::collections::HashSet;
        let mut cls_out = ClassificationOutput::<u8, f32>::new(30u8);
        let mut images = vec!["india.jpg", "germany.png", "iran.jpg", "canada.png", "japan.jpg"];
        images.sort();
        for img in &images {
            let v = vec![1f32; 30];
            cls_out.add(img, v).unwrap();
        }
        let list_of_images = cls_out.list_images();
        let mut lhs = HashSet::<&str>::with_capacity(list_of_images.len());
        for item in list_of_images {
            lhs.insert(item);
        }

        let mut rhs = HashSet::<&str>::with_capacity(images.len());
        for item in images {
            rhs.insert(item);
        }
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn classification_test_classification_output_is_empty() {
        let cls_out = ClassificationOutput::<u8, f64>::new(20u8);
        assert_eq!(cls_out.is_empty(), true);
    }

    #[test]
    fn classification_test_confidence_for_image() {
        let mut cls_out = ClassificationOutput::<u32, f32>::new(1000u32);
        let rand_iter = rand::thread_rng().gen_range(0usize..500usize);
        let mut test_image = String::new();
        let mut test_vec = Vec::<f32>::new();
        for i in 0usize..500usize {
            let rand_name: String = (0..50).map(|_| rand::random::<u8>() as char).collect();
            let rand_vec = vec![rand::random::<f32>(); 1000usize];
            if i == rand_iter {
                test_vec = rand_vec.clone();
                test_image = rand_name.clone();
            }
            cls_out.add(&rand_name, rand_vec).unwrap();
        }

        for (lhs, rhs) in (*(cls_out.confidence_for_image(&test_image).unwrap())).iter().zip(test_vec.iter()) {
            approx_eq!(f32, *lhs, *rhs, ulps=5);
        }
    }

    #[test]
    fn classification_test_topk_for_image() {
        let mut cls_out = ClassificationOutput::<u32, f32>::new(1000u32);
        let rand_iter = rand::thread_rng().gen_range(0usize..500usize);
        let mut test_image = String::new();
        for i in 0usize..500usize {
            let rand_name: String = (0..50).map(|_| rand::random::<u8>() as char).collect();
            let rand_vec: Vec<f32>;
            if i == rand_iter {
                test_image = rand_name.clone();
                rand_vec = (0..1000).map(|x| x as f32).collect();
            } else {
                rand_vec = vec![rand::random::<f32>(); 1000usize];
            }
            cls_out.add(&rand_name, rand_vec).unwrap();
        }

        let topk_ind = cls_out.topk_for_image(&test_image, 3usize).unwrap();
        assert_eq!(topk_ind, vec![999usize, 998usize, 997usize])
    }
}

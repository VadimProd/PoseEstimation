from PoseEstimator import PoseEstimator


if __name__ == '__main__':
    estimator = PoseEstimator(
        config_file='../configs/ae_hrnet-w32_8xb24-300e_coco-512x512.py',
        checkpoint_file='../configs/hrnet_w32_coco_512x512-bcb8c247_20200816.pth',
        method='bottomup'
    )

    # estimator = PoseEstimator(
    #     config_file='/mmpose/data/configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
    #     checkpoint_file='/mmpose/data/configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth',
    #     method='topdown'
    # )

    estimator.process(
        input_path='../test_data/video/video2.mp4',
        output_path='../test_data/out/video2_out.mp4'
    )

    # estimator.process(
    #     input_path='/mmpose/data/test_data/images/image4.jpg',
    #     output_path='/mmpose/data/test_data/out/image4_out.jpg'
    # )
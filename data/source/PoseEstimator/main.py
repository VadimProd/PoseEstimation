from data.source.PoseEstimator.PoseEstimator import PoseEstimator
import argparse


def main():
    parser = argparse.ArgumentParser(description="Pose Estimation Tool")
    parser.add_argument("--method", choices=["topdown", "bottomup", "hybrid"], default="bottomup")
    parser.add_argument("--input", type=str, required=True, help="Input image/video path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    args = parser.parse_args()

    estimator = PoseEstimator(method=args.method)
    estimator.process(input_path=args.input, output_path=args.output)

if __name__ == '__main__':
    main()
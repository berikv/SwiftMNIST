import Foundation

extension MNISTDataset {
    static let training: MNISTDataset = {
        let labelPath = Bundle.main.path(forResource: "train-labels", ofType: "idx1-ubyte")!
        let imagePath = Bundle.main.path(forResource: "train-images", ofType: "idx3-ubyte")!
        return try! MNISTDataset(labelPath: labelPath, imagePath: imagePath)
    }()

    static let validation: MNISTDataset = {
        let labelPath = Bundle.main.path(forResource: "t10k-labels", ofType: "idx1-ubyte")!
        let imagePath = Bundle.main.path(forResource: "t10k-images", ofType: "idx3-ubyte")!
        return try! MNISTDataset(labelPath: labelPath, imagePath: imagePath)
    }()
}

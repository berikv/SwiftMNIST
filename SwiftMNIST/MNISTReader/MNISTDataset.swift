import Foundation

struct MNISTDataset {
    let sampleCount: Int
    let rowCount: Int
    let columnCount: Int
    let labelData: Data
    let imageData: Data

    let labelDataHeader = 8
    let imageDataHeader = 16

    var width: Int { columnCount }
    var height: Int { rowCount }

    init(labelPath: String, imagePath: String) throws {
        let labelData = try Data(contentsOf: URL(fileURLWithPath: labelPath))
        let labelMagic = labelData.readUInt32()
        let labelCount = labelData.readUInt32(offset: 4)

        guard labelMagic == 2049 else { fatalError("label file has unexpected header") }
        guard labelData.count == labelDataHeader + Int(labelCount) else { fatalError() }

        let imageData = try Data(contentsOf: URL(fileURLWithPath: imagePath))
        let imageMagic = imageData.readUInt32()
        let imageCount = imageData.readUInt32(offset: 4)
        let rowCount = imageData.readUInt32(offset: 8)
        let columnCount = imageData.readUInt32(offset: 12)

        guard imageMagic == 2051 else { fatalError("image file has unexpected header") }
        let totalExpectedImageDataCount = imageDataHeader + Int(imageCount * rowCount * columnCount)
        guard imageData.count == totalExpectedImageDataCount else { fatalError() }
        guard labelCount == imageCount else { fatalError() }

        self.sampleCount = Int(labelCount)
        self.rowCount = Int(rowCount)
        self.columnCount = Int(columnCount)

        self.labelData = labelData
        self.imageData = imageData
    }
}

private extension Data {
    func readUInt32(offset: Int = 0) -> UInt32 {
        UInt32(bigEndian: withUnsafeBytes {
            $0.load(fromByteOffset: offset, as: UInt32.self)
        })
    }
}

extension MNISTDataset: RandomAccessCollection {
    typealias Index = Int
    typealias Element = MNISTSample

    var startIndex: Int { 0 }
    var endIndex: Int { sampleCount }

    subscript(index: Index) -> Iterator.Element {
        get {
            let imageStart = imageDataHeader + index * rowCount * columnCount
            let imageEnd = imageDataHeader + (index + 1) * rowCount * columnCount
            let imageData = imageData[imageStart..<imageEnd]
            let label = labelData[labelDataHeader + index]

            return MNISTSample(label: label, image: imageData)
        }
    }

    func index(after i: Index) -> Index {
        return i + 1
    }
}

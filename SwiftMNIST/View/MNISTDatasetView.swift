import SwiftUI
import CoreGraphics

struct MNISTDatasetView: View {
    @Binding var dataset: MNISTDataset
    let columns = Array(repeating: GridItem(.flexible(minimum: 25, maximum: 50)), count: 8)
    var body: some View {
        ScrollView(.vertical) {
            LazyVGrid(columns: columns) {
                ForEach(dataset, id: \.self) { sample in
                    Text("\(sample.label)")
                    Image(sample: sample, width: dataset.width, height: dataset.height)
                }
            }
        }
    }
}

extension Image {
    init(sample: MNISTSample, width: Int, height: Int) {
        let size = NSSize(width: width, height: height)
        self.init(nsImage: .from(bytePerPixelGrayscaleData: sample.image, size: size))
    }
}

extension NSImage {
    static func from(bytePerPixelGrayscaleData imageData: Data, size: NSSize) -> NSImage {
        // ImageData won't be mutated, but CGContext requires a UnsafeMutableRawPointer.
        // That is probably a mistake in the Swift API for CGContext(data: ...)
        var imageData = imageData
        let context = imageData.withUnsafeMutableBytes { pointer in
             CGContext(
                data: pointer.baseAddress!,
                width: Int(size.width),
                height: Int(size.height),
                bitsPerComponent: 8,
                bytesPerRow: Int(size.width),
                space: CGColorSpace(name: CGColorSpace.linearGray)!,
                bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue)!
        }

        return NSImage(cgImage: context.makeImage()!, size: size)
    }
}

#Preview {
    MNISTDatasetView(dataset: .constant(MNISTDataset.training))
}

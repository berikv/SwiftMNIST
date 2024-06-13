import SwiftUI

extension Image {
    init(bitmap: Data, width: Int, height: Int) {
        self.init(nsImage: .from(bytePerPixelGrayscaleData: bitmap, width: width, height: height))
    }
}

extension NSImage {
    static func from(bytePerPixelGrayscaleData imageData: Data, width: Int, height: Int) -> NSImage {
        // ImageData won't be mutated, but CGContext requires a UnsafeMutableRawPointer.
        // That is probably a mistake in the Swift API for CGContext(data: ...)
        var imageData = imageData
        let context = imageData.withUnsafeMutableBytes { pointer in
             CGContext(
                data: pointer.baseAddress!,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width,
                space: CGColorSpace(name: CGColorSpace.linearGray)!,
                bitmapInfo: CGBitmapInfo.byteOrderDefault.rawValue)!
        }

        let image = NSImage(cgImage: context.makeImage()!, size: NSZeroSize)
        return image
    }
}

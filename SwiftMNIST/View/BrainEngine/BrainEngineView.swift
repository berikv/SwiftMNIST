import SwiftUI

struct BrainEngineView: View {
    @State private var selectedTab = "training"

    var body: some View {
        TabView(selection: $selectedTab) {
            BrainEngineTraingingView(goToValidation: { selectedTab = "validation" })
                .tabItem { Label("Training", systemImage: "book.circle") }
                .tag("training")
            BrainEngineValidationView()
                .tabItem { Label("Validation", systemImage: "square.and.pencil") }
                .tag("validation")
        }
    }
}

#Preview {
    BrainEngineView()
}

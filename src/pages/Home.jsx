import React from "react";
// Import your existing components
// Example:
// import SettingsPanel from "../components/SettingsPanel.jsx";
// import ImageDisplay from "../components/ImageDisplay.jsx";
// import ControlButtons from "../components/ControlButtons.jsx";
// import ResultsTable from "../components/ResultsTable.jsx";

export default function Home() {
  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Place your existing main UI here */}
      {/* <SettingsPanel ... /> */}
      {/* <ImageDisplay ... /> */}
      {/* <ControlButtons ... /> */}
      {/* <ResultsTable ... /> */}
      <div className="bg-gray-800 rounded-xl shadow-lg p-4">
        <p className="text-gray-300">
          Welcome to SHAI — use the controls to open an image or capture live.
        </p>
      </div>
    </div>
  );
}
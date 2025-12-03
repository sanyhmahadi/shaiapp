export default function About() {
  return (
    <div className="bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 max-w-3xl mx-auto">
      <h2 className="text-xl sm:text-2xl font-bold text-gray-100 mb-6 text-center">
        About Us
      </h2>

      {/* English Section */}
      <div className="mb-10">
        <h3 className="text-lg sm:text-xl font-semibold text-violet-400 mb-2">
          🌍 About Us (English)
        </h3>
        <div className="text-gray-300 text-justify space-y-3">
          <p>
            <strong>SHAI — South Harbour AI App for Sustainable Recycling</strong>
          </p>
          <p>
            SHAI is a web-based application born from a collaboration between students and the South Harbor waste management facility in Copenhagen. Our mission is to empower citizens to participate more actively in recycling through intuitive, AI-powered tools.
          </p>
          <p>
            Built on the principles of <em>design thinking</em>, <em>user-centered design</em>, and <em>education for sustainable development (ESD)</em>, SHAI combines object detection with motivational strategies to make recycling easier, smarter, and more engaging.
          </p>
          <p>
            We believe that even small local actions can inspire global change. By bridging the gap between technology and human behavior, SHAI helps users learn, participate, and stay committed to a greener future.
          </p>
        </div>
      </div>

      {/* Danish Section */}
      <div>
        <h3 className="text-lg sm:text-xl font-semibold text-green-400 mb-2">
          🌱 Om Os (Dansk)
        </h3>
        <div className="text-gray-300 text-justify space-y-3">
          <p>
            <strong>SHAI — South Harbour AI App til Bæredygtig Genbrug</strong>
          </p>
          <p>
            SHAI er en webbaseret applikation udviklet i samarbejde mellem studerende og genbrugsstationen i Sydhavn, København. Vores mål er at styrke borgernes deltagelse i genbrug gennem intuitive og AI-drevne værktøjer.
          </p>
          <p>
            Appen bygger på principperne om <em>design thinking</em>, <em>brugercentreret design</em> og <em>uddannelse for bæredygtig udvikling (ESD)</em>. SHAI kombinerer objektgenkendelse med motiverende strategier for at gøre genbrug lettere, smartere og mere engagerende.
          </p>
          <p>
            Vi tror på, at selv små lokale handlinger kan inspirere global forandring. Ved at bygge bro mellem teknologi og menneskelig adfærd hjælper SHAI brugerne med at lære, deltage og forblive engagerede i en grønnere fremtid.
          </p>
        </div>
      </div>
    </div>
  );
}
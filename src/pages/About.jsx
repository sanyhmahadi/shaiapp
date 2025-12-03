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
          Hi, and welcome to the <strong>SHAI app</strong> — a student project from 
          <em> Aalborg University (Copenhagen Campus)</em>. This is a prototype for a tool 
          that will help you scan your household waste before recycling.
          </p>
          <p>
          Please take a picture of the trash and click the icon to see which container it belongs to. 
          If it doesn’t work, please try again from another angle. We would appreciate any feedback 
          and recommendations as the project is still in development. Thank you for your effort and 
          your patience.
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
          <p>
          <strong>This project was done by:</strong><br />
          • Mathias Lind — Co-ordination<br />
          • Anders Kassa Häggquist — Co-ordination<br />
          • Mahadi Hasan Sany — Prototype Development & Co-ordination<br />
          </p>
          <p>
          <strong>Contact:</strong><br />
          Mathias — <a 
            href="mailto:mlyngs24@student.aau.dk" 
            className="text-green-400 hover:underline"
          >
            mlyngs24@student.aau.dk
          </a>
          <br />
          Anders — <a 
            href="mailto:ahaggq24@student.aau.dk" 
            className="text-green-400 hover:underline"
          >
            ahaggq24@student.aau.dk
          </a>
          <br />
          Sany — <a 
            href="mailto:msany24@student.aau.dk" 
            className="text-green-400 hover:underline"
          >
            msany24@student.aau.dk
          </a> | <a 
            href="https://github.com/sanyhmahadi" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-green-400 hover:underline"
          >
            GitHub Profile
          </a>
          </p>

        </div>
      </div>

      {/* Danish Section */}
<div className="mb-10">
  <h3 className="text-lg sm:text-xl font-semibold text-green-400 mb-2">
    🌱 Om Os (Dansk)
  </h3>
  <div className="text-gray-300 text-justify space-y-3">
    <p>
      <strong>SHAI — South Harbour AI App til Bæredygtig Genbrug</strong>
    </p>
    <p>
      Hej, og velkommen til <strong>SHAI appen</strong> — et studieprojekt fra 
      <em> Aalborg Universitet (København Campus)</em>. Dette er en prototype på et værktøj, 
      der hjælper dig med at scanne dit husholdningsaffald før genbrug.
    </p>
    <p>
      Tag venligst et billede af affaldet og klik på ikonet for at se, hvilken container det hører til. 
      Hvis det ikke virker, prøv venligst igen fra en anden vinkel. Vi sætter pris på al feedback 
      og anbefalinger, da projektet stadig er under udvikling. Tak for din indsats og din tålmodighed.
    </p>
    <p>
      SHAI er en webbaseret applikation udviklet i samarbejde mellem studerende og genbrugsstationen i Sydhavn, København. 
      Vores mission er at styrke borgernes deltagelse i genbrug gennem intuitive, AI-drevne værktøjer.
    </p>
    <p>
      Appen bygger på principperne om <em>design thinking</em>, <em>brugercentreret design</em> og 
      <em> uddannelse for bæredygtig udvikling (ESD)</em>. SHAI kombinerer objektgenkendelse med motiverende strategier 
      for at gøre genbrug lettere, smartere og mere engagerende.
    </p>
    <p>
      Vi tror på, at selv små lokale handlinger kan inspirere global forandring. Ved at bygge bro mellem teknologi og 
      menneskelig adfærd hjælper SHAI brugerne med at lære, deltage og forblive engagerede i en grønnere fremtid.
    </p>
    <p>
      <strong>Dette projekt er lavet af:</strong><br />
      • Mathias Lind — Koordination<br />
      • Anders Kassa Häggquist — Koordination<br />
      • Mahadi Hasan Sany — Prototypeudvikling & Koordination<br />
    </p>
    <p>
      <strong>Kontakt:</strong><br />
      Mathias — <a 
        href="mailto:mlyngs24@student.aau.dk" 
        className="text-green-400 hover:underline"
      >
        mlyngs24@student.aau.dk
      </a>
      <br />
      Anders — <a 
        href="mailto:ahaggq24@student.aau.dk" 
        className="text-green-400 hover:underline"
      >
        ahaggq24@student.aau.dk
      </a>
      <br />
      Sany — 
          <a 
            href="mailto:msany24@student.aau.dk" 
            className="text-green-400 hover:underline"
          >
            msany24@student.aau.dk
          </a> | <a 
            href="https://github.com/sanyhmahadi" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-green-400 hover:underline"
          >
            GitHub Profile
          </a>
    </p>
  </div>
</div>
    </div>
  );
}
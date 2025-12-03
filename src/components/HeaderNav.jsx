import { Link, useLocation } from "react-router-dom";

export default function HeaderNav() {
  const { pathname } = useLocation();

  const base = "inline-flex items-center px-3 py-2 rounded-md text-sm font-medium transition-all";
  const active = "bg-violet-600 text-white";
  const inactive = "bg-gray-700 text-white hover:bg-gray-600";

  return (
    <header className="bg-gray-800 shadow-lg mb-4 sm:mb-6">
      <div className="max-w-7xl mx-auto px-3 sm:px-4 py-3 flex items-center justify-between">
        <div className="text-lg sm:text-xl font-bold">Welcome to SHAI App</div>
        <nav className="flex items-center gap-2">
          <Link
            to="/"
            className={`${base} ${pathname === "/" ? active : inactive}`}
          >
            Home
          </Link>
          
          <Link
            to="/calendar"
            className={`${base} ${pathname === "/calendar" ? active : inactive}`}
          >
            Calendar
          </Link>

          <Link
            to="/about"
            className={`${base} ${pathname === "/about" ? active : inactive}`}
          >
            About Us
          </Link>
        </nav>
      </div>
    </header>
  );
}
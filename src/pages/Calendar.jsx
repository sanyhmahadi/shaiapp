import React, { useMemo, useState } from "react";

const monthNames = [
  "January","February","March","April","May","June",
  "July","August","September","October","November","December",
];
const weekdayNames = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];

const initialEvents = [
  // Example events (YYYY-MM-DD)
  { date: "2025-12-05", title: "Re-Use Workshop" },
  { date: "2025-12-08", title: "Live Articraft Making" },
  { date: "2025-12-15", title: "Deploy new bin images" },
];

export default function Calendar() {
  const [current, setCurrent] = useState(new Date());

  const year = current.getFullYear();
  const month = current.getMonth();

  const firstOfMonth = new Date(year, month, 1);
  const startWeekday = firstOfMonth.getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const todayStr = toISO(new Date());

  const eventsByDay = useMemo(() => {
    const map = {};
    for (const ev of initialEvents) {
      if (!map[ev.date]) map[ev.date] = [];
      map[ev.date].push(ev);
    }
    return map;
  }, []);

  const cells = useMemo(() => {
    const arr = [];
    for (let i = 0; i < startWeekday; i++) arr.push(null);
    for (let day = 1; day <= daysInMonth; day++) {
      arr.push(new Date(year, month, day));
    }
    return arr;
  }, [year, month, startWeekday, daysInMonth]);

  const prevMonth = () => setCurrent(new Date(year, month - 1, 1));
  const nextMonth = () => setCurrent(new Date(year, month + 1, 1));
  const thisMonth = () => setCurrent(new Date());

  return (
    <div className="bg-gray-800 rounded-xl shadow-lg p-4 sm:p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xl sm:text-2xl font-bold text-gray-100">
          {monthNames[month]} {year}
        </h2>
        <div className="flex gap-2">
          <button className="btn-secondary" onClick={prevMonth}>Prev</button>
          <button className="btn-secondary" onClick={thisMonth}>Today</button>
          <button className="btn-secondary" onClick={nextMonth}>Next</button>
        </div>
      </div>

      <div className="grid grid-cols-7 gap-2 mb-2">
        {weekdayNames.map((w) => (
          <div key={w} className="text-center text-xs sm:text-sm text-gray-300">
            {w}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-7 gap-2">
        {cells.map((dateObj, i) => {
          if (!dateObj) {
            return <div key={`empty-${i}`} className="h-20 sm:h-24 bg-gray-700 rounded-md" />;
          }
          const iso = toISO(dateObj);
          const isToday = iso === todayStr;
          const evs = eventsByDay[iso] || [];

          return (
            <div
              key={iso}
              className={`h-20 sm:h-24 rounded-md p-2 border ${
                isToday ? "border-violet-500" : "border-gray-700"
              } bg-gray-700`}
            >
              <div className="flex justify-between items-center">
                <span className="text-sm font-semibold text-gray-100">
                  {dateObj.getDate()}
                </span>
                {evs.length > 0 && (
                  <span className="text-[10px] px-1 py-0.5 rounded bg-violet-600 text-white">
                    {evs.length} event{evs.length > 1 ? "s" : ""}
                  </span>
                )}
              </div>

              <div className="mt-1 space-y-1 overflow-y-auto max-h-14">
                {evs.map((ev, idx) => (
                  <div
                    key={`${iso}-${idx}`}
                    className="text-[11px] sm:text-xs bg-gray-800 text-gray-200 rounded px-1 py-0.5"
                    title={ev.title}
                  >
                    • {ev.title}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function toISO(d) {
  const y = d.getFullYear();
  const m = (`0${d.getMonth() + 1}`).slice(-2);
  const day = (`0${d.getDate()}`).slice(-2);
  return `${y}-${m}-${day}`;
}
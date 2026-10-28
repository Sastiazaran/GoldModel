import fs from "node:fs";
import path from "node:path";
import { Tablero } from "../components/Tablero";
import type { Reporte, Snapshot } from "../lib/types";

function leer<T>(nombre: string, vacio: T): T {
  const ruta = path.join(process.cwd(), "data", nombre);
  if (!fs.existsSync(ruta)) return vacio;
  return JSON.parse(fs.readFileSync(ruta, "utf8")) as T;
}

export default function Page() {
  const reportes = leer<Reporte[]>("reportes.json", []);
  const snapshot = leer<Snapshot>("snapshot.json", {
    generado_en: null,
    aviso: "No es asesoría financiera.",
    modelo: null,
  });
  return (
    <main>
      <Tablero reportes={reportes} snapshot={snapshot} />
    </main>
  );
}

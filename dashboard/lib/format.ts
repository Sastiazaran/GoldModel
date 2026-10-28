const GRAMOS_POR_ONZA = 31.1034768;

export function fechaLarga(isoDia: string): string {
  const fecha = fechaUtc(isoDia);
  const texto = new Intl.DateTimeFormat("es-MX", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(fecha);
  return texto.charAt(0).toUpperCase() + texto.slice(1);
}

export function fechaCorta(isoDia: string): string {
  return new Intl.DateTimeFormat("es-MX", {
    day: "numeric",
    month: "short",
    timeZone: "UTC",
  })
    .format(fechaUtc(isoDia))
    .replace(".", "");
}

export function fechaMedia(isoDia: string): string {
  return new Intl.DateTimeFormat("es-MX", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  })
    .format(fechaUtc(isoDia))
    .replace(".", "");
}

export function instante(iso: string): string {
  const fecha = new Date(iso);
  if (Number.isNaN(fecha.getTime())) return iso;
  return new Intl.DateTimeFormat("es-MX", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "America/Mexico_City",
  }).format(fecha);
}

function fechaUtc(isoDia: string): Date {
  const [anio, mes, dia] = isoDia.split("-").map(Number);
  return new Date(Date.UTC(anio, mes - 1, dia));
}

export function numero(valor: number, decimales: number): string {
  return new Intl.NumberFormat("es-MX", {
    minimumFractionDigits: decimales,
    maximumFractionDigits: decimales,
  }).format(valor);
}

export function conSigno(valor: number, decimales: number): string {
  const cuerpo = numero(Math.abs(valor), decimales);
  if (valor > 0) return `+${cuerpo}`;
  if (valor < 0) return `−${cuerpo}`;
  return cuerpo;
}

export function usd(valor: number, decimales = 2): string {
  return `$${numero(valor, decimales)}`;
}

export function pesos(valor: number): string {
  return `$${numero(valor, 0)}`;
}

export function puntos(valor: number | null, decimales = 2): string {
  if (valor == null || !Number.isFinite(valor)) return "Sin dato";
  return `${conSigno(valor, decimales)}%`;
}

export function fraccion(valor: number | null): string {
  if (valor == null || !Number.isFinite(valor)) return "Sin dato";
  return `${numero(valor * 100, 1)}%`;
}

export function r2(valor: number | null): string {
  if (valor == null || !Number.isFinite(valor)) return "Sin dato";
  return conSigno(valor, 3);
}

export function cuentaGramos(precioUsd: number, tipoCambio: number): number {
  return (precioUsd * tipoCambio) / GRAMOS_POR_ONZA;
}

export const GRAMOS = GRAMOS_POR_ONZA;

export function textoDireccion(direccion: "alza" | "baja" | "sin_cambio"): string {
  if (direccion === "alza") return "Apunta al alza";
  if (direccion === "baja") return "Apunta a la baja";
  return "Apunta a un cambio de cero";
}

export function textoBrecha(brecha: number): string {
  const magnitud = numero(Math.abs(brecha), 1);
  if (brecha > 0.05) return `${magnitud}% por encima del valor justo`;
  if (brecha < -0.05) return `${magnitud}% por debajo del valor justo`;
  return "pegado al valor justo";
}

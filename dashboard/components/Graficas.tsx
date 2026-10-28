"use client";

import { useId, useState } from "react";
import { fechaCorta, numero, usd } from "../lib/format";
import type { Punto, Pronostico } from "../lib/types";

const ANCHO = 960;
const ALTO = 420;
const PAD = { l: 72, r: 28, t: 22, b: 42 };

type Marca = {
  horizonte: number;
  nivel: number | null;
  direccion: Pronostico["direccion"];
};

type Props = {
  puntos: Punto[];
  marcas: Marca[];
  mostrarAmpliado: boolean;
  mostrarNuevo: boolean;
  desde: string;
  nombresRegimen?: string[];
};

const CLASE_REGIMEN = ["linea-reg-0", "linea-reg-1", "linea-reg-2"];

export function GraficaPrecio({ puntos, marcas, mostrarAmpliado, mostrarNuevo, desde, nombresRegimen = [] }: Props) {
  const tituloId = useId();
  const [hover, setHover] = useState<number | null>(null);
  const n = puntos.length;
  if (n < 2) {
    return <p className="nota">No hay historia suficiente para la gráfica.</p>;
  }

  const xMax = n - 1 + 20;
  const valores: number[] = [];
  for (const punto of puntos) {
    if (punto.precio != null) valores.push(punto.precio);
    if (punto.valor_justo != null) valores.push(punto.valor_justo);
    if (mostrarAmpliado && punto.valor_ampliado != null) valores.push(punto.valor_ampliado);
    if (punto.pronostico_20 != null) valores.push(punto.pronostico_20);
    if (punto.intervalo_bajo != null) valores.push(punto.intervalo_bajo);
    if (punto.intervalo_alto != null) valores.push(punto.intervalo_alto);
    if (mostrarNuevo && punto.valor_nuevo != null) valores.push(punto.valor_nuevo);
  }
  for (const marca of marcas) {
    if (marca.nivel != null) valores.push(marca.nivel);
  }
  const minDato = Math.min(...valores);
  const maxDato = Math.max(...valores);
  const holgura = (maxDato - minDato) * 0.06 || 1;
  const min = minDato - holgura;
  const max = maxDato + holgura;

  const xDe = (indice: number) => PAD.l + (indice / xMax) * (ANCHO - PAD.l - PAD.r);
  const yDe = (valor: number) => PAD.t + (1 - (valor - min) / (max - min)) * (ALTO - PAD.t - PAD.b);

  const trazos = (clave: "precio" | "valor_justo" | "valor_ampliado" | "valor_nuevo" | "pronostico_20") => {
    const trozos: string[] = [];
    let actual: string[] = [];
    puntos.forEach((punto, indice) => {
      const valor = punto[clave];
      if (valor == null) {
        if (actual.length > 1) trozos.push(actual.join(" "));
        actual = [];
        return;
      }
      actual.push(`${xDe(indice).toFixed(1)},${yDe(valor).toFixed(1)}`);
    });
    if (actual.length > 1) trozos.push(actual.join(" "));
    return trozos;
  };

  const precioPorRegimen = () => {
    const trozos: { clase: string; puntos: string }[] = [];
    let actual: string[] = [];
    let clase = "";
    const cerrar = () => {
      if (actual.length > 1) trozos.push({ clase, puntos: actual.join(" ") });
      actual = [];
    };
    puntos.forEach((punto, indice) => {
      if (punto.precio == null) {
        cerrar();
        clase = "";
        return;
      }
      const siguiente = punto.regimen == null ? "linea-precio" : CLASE_REGIMEN[punto.regimen] ?? "linea-precio";
      const coord = `${xDe(indice).toFixed(1)},${yDe(punto.precio).toFixed(1)}`;
      if (clase && siguiente !== clase) {
        actual.push(coord);
        cerrar();
      }
      clase = siguiente;
      actual.push(coord);
    });
    cerrar();
    return trozos;
  };

  const bandaIntervalo = () => {
    const altos: string[] = [];
    const bajos: string[] = [];
    puntos.forEach((punto, indice) => {
      if (punto.intervalo_bajo == null || punto.intervalo_alto == null) return;
      const x = xDe(indice).toFixed(1);
      altos.push(`${x},${yDe(punto.intervalo_alto).toFixed(1)}`);
      bajos.push(`${x},${yDe(punto.intervalo_bajo).toFixed(1)}`);
    });
    if (altos.length < 2) return null;
    return [...altos, ...bajos.reverse()].join(" ");
  };
  const banda = bandaIntervalo();

  const ticks = [0, 0.33, 0.66, 1].map((parte) => min + (max - min) * (1 - parte));
  const ultimo = puntos[n - 1];
  const primero = puntos[0];
  const medio = puntos[Math.floor(n / 2)];
  const activo = hover != null ? puntos[hover] : ultimo;
  const marca5 = marcas.find((marca) => marca.horizonte === 5);
  const marca20 = marcas.find((marca) => marca.horizonte === 20);
  const resumen = [
    `Precio del futuro y valor justo desde ${desde}.`,
    ultimo.precio != null ? `Último cierre ${usd(ultimo.precio)}.` : "",
    ultimo.valor_justo != null ? `Valor justo ${usd(ultimo.valor_justo)}.` : "",
    marca5?.nivel != null ? `Pronóstico a 5 días: ${usd(marca5.nivel)}.` : "",
    marca20?.nivel != null ? `Pronóstico a 20 días: ${usd(marca20.nivel)}.` : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="grafica">
      <svg
        viewBox={`0 0 ${ANCHO} ${ALTO}`}
        role="img"
        aria-labelledby={tituloId}
        onPointerMove={(evento) => {
          const rect = evento.currentTarget.getBoundingClientRect();
          const viewX = ((evento.clientX - rect.left) / rect.width) * ANCHO;
          const indice = Math.round(((viewX - PAD.l) / (ANCHO - PAD.l - PAD.r)) * xMax);
          setHover(Math.max(0, Math.min(n - 1, indice)));
        }}
        onPointerLeave={() => setHover(null)}
      >
        <title id={tituloId}>{resumen}</title>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="guia" x1={PAD.l} x2={ANCHO - PAD.r} y1={yDe(tick)} y2={yDe(tick)} />
            <text className="eje" x={PAD.l - 10} y={yDe(tick) + 4} textAnchor="end">
              {numero(tick, 0)}
            </text>
          </g>
        ))}
        <line className="hoy" x1={xDe(n - 1)} x2={xDe(n - 1)} y1={PAD.t} y2={ALTO - PAD.b} />
        {banda && <polygon className="banda-intervalo" points={banda} />}
        {trazos("pronostico_20").map((puntosLinea) => (
          <polyline key={puntosLinea.slice(0, 24)} className="linea-pronostico" points={puntosLinea} />
        ))}
        {mostrarAmpliado &&
          trazos("valor_ampliado").map((puntosLinea) => (
            <polyline key={puntosLinea.slice(0, 24)} className="linea-ampliada" points={puntosLinea} />
          ))}
        {mostrarNuevo &&
          trazos("valor_nuevo").map((puntosLinea) => (
            <polyline key={puntosLinea.slice(0, 24)} className="linea-nueva" points={puntosLinea} />
          ))}
        {trazos("valor_justo").map((puntosLinea) => (
          <polyline key={puntosLinea.slice(0, 24)} className="linea-justo" points={puntosLinea} />
        ))}
        {precioPorRegimen().map((trazo, indice) => (
          <polyline key={`${trazo.clase}-${indice}`} className={trazo.clase} points={trazo.puntos} />
        ))}
        {ultimo.precio != null &&
          marcas.map((marca) => {
            if (marca.nivel == null || ultimo.precio == null) return null;
            const x = xDe(n - 1 + marca.horizonte);
            const y = yDe(marca.nivel);
            return (
              <g key={marca.horizonte}>
                <line
                  className={marca.direccion === "baja" ? "puente puente-baja" : "puente puente-alza"}
                  x1={xDe(n - 1)}
                  y1={yDe(ultimo.precio)}
                  x2={x}
                  y2={y}
                />
                <circle className={marca.direccion === "baja" ? "marca marca-baja" : "marca marca-alza"} cx={x} cy={y} r="5" />
                <text className="eje" x={x} y={ALTO - 8} textAnchor="middle">
                  +{marca.horizonte}d
                </text>
              </g>
            );
          })}
        {hover != null && activo.precio != null && (
          <line className="cursor" x1={xDe(hover)} x2={xDe(hover)} y1={PAD.t} y2={ALTO - PAD.b} />
        )}
        <text className="eje" x={xDe(0)} y={ALTO - 8} textAnchor="start">
          {fechaCorta(primero.fecha)}
        </text>
        <text className="eje" x={xDe(Math.floor(n / 2))} y={ALTO - 8} textAnchor="middle">
          {fechaCorta(medio.fecha)}
        </text>
        <text className="eje" x={xDe(n - 1)} y={ALTO - 22} textAnchor="middle">
          {fechaCorta(ultimo.fecha)}
        </text>
      </svg>
      <div className="leyenda" aria-hidden="true">
        <span><i className="muestra precio" /> Precio GC=F, color del régimen</span>
        {nombresRegimen.map((nombre, indice) => (
          <span key={nombre}>
            <i className={`muestra reg-${indice}`} /> {nombre}
          </span>
        ))}
        <span><i className="muestra justo" /> Valor justo titular</span>
        <span><i className="muestra intervalo" /> Banda de 20 días</span>
        <span><i className="muestra pronostico" /> Pronóstico de 20 días, en el día que vence</span>
        {mostrarAmpliado && <span><i className="muestra ampliado" /> Valor justo ampliado</span>}
        {mostrarNuevo && <span><i className="muestra nuevo" /> Valor justo Kalman</span>}
      </div>
      <p className="lectura-grafica">
        {fechaCorta(activo.fecha)}
        {activo.precio != null ? ` · precio ${usd(activo.precio)}` : ""}
        {activo.valor_justo != null ? ` · valor justo ${usd(activo.valor_justo)}` : ""}
        {activo.pronostico_20 != null ? ` · lo que apuntaba el pronóstico de 20 días ${usd(activo.pronostico_20)}` : ""}
      </p>
    </div>
  );
}

export function GraficaAcierto({ serie }: { serie: { fecha: string; tasa: number }[] }) {
  if (serie.length < 2) return null;
  const ancho = 320;
  const alto = 88;
  const pad = 8;
  const puntos = serie
    .map((punto, indice) => {
      const x = pad + (indice / (serie.length - 1)) * (ancho - pad * 2);
      const y = pad + (1 - punto.tasa) * (alto - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const yMedia = pad + 0.5 * (alto - pad * 2);
  const ultima = serie[serie.length - 1];
  return (
    <svg className="chispa" viewBox={`0 0 ${ancho} ${alto}`} role="img" aria-label={`Acierto acumulado, termina en ${Math.round(ultima.tasa * 100)} por ciento.`}>
      <line className="guia" x1={pad} x2={ancho - pad} y1={yMedia} y2={yMedia} />
      <polyline className="linea-acierto" points={puntos} />
    </svg>
  );
}

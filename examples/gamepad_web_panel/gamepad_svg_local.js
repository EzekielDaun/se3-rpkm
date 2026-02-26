"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

function appendChild(parent, child) {
  if (Array.isArray(child)) {
    for (const nested of child) {
      appendChild(parent, nested);
    }
    return;
  }

  if (
    child === undefined ||
    child === null ||
    child === false ||
    child === true
  ) {
    return;
  }

  if (child instanceof Node) {
    parent.appendChild(child);
    return;
  }

  parent.appendChild(document.createTextNode(String(child)));
}

function camelToKebab(name) {
  return name.replace(/[A-Z]/g, (char) => `-${char.toLowerCase()}`);
}

function createElement(type, props) {
  const element = document.createElementNS(SVG_NS, type);

  if (props !== undefined && props !== null) {
    for (const [key, value] of Object.entries(props)) {
      if (key === "children") {
        continue;
      }
      if (value === undefined || value === null || value === false) {
        continue;
      }
      const attrName = key === "viewBox" ? "viewBox" : camelToKebab(key);
      if (value === true) {
        element.setAttribute(attrName, "");
        continue;
      }
      element.setAttribute(attrName, String(value));
    }

    appendChild(element, props.children);
  }

  return element;
}

const c = {
  jsx: createElement,
  jsxs: createElement,
};

const THEME_PALETTES = {
  light: {
    controllerArtLight: "hsl(210,50%,85%)",
    controllerBodyFill: "#ffffff",
  },
  dark: {
    controllerArtLight: "hsl(204,44%,64%)",
    controllerBodyFill: "#c7d3e4",
  },
};
const fe = { ...THEME_PALETTES.light };
let currentTheme = "light";
let currentGamepad = null;

function applyTheme(themeName) {
  const theme = themeName === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", theme);
  if (theme === currentTheme) {
    return;
  }

  currentTheme = theme;
  const palette = THEME_PALETTES[theme];
  fe.controllerArtLight = palette.controllerArtLight;
  fe.controllerBodyFill = palette.controllerBodyFill;
  if (currentGamepad !== null) {
    renderGamepad(currentGamepad);
  }
}

window.setGamepadTheme = function (themeName) {
  applyTheme(themeName);
};

function re(e) {
  return typeof e == "number" ? e : e.value;
}
function A5({ gamepad: e, width: t }) {
  const r = fe.controllerArtLight,
    a = 3,
    i = Math.max(
      0,
      -0.2 + 1.2 * Math.sqrt(Math.pow(e.axes[0], 2) + Math.pow(e.axes[1], 2)),
    ),
    o = Math.max(
      0,
      -0.2 + 1.2 * Math.sqrt(Math.pow(e.axes[2], 2) + Math.pow(e.axes[3], 2)),
    );
  var s = /playstation|dualsense|dualshock/i.test(e.id) || /054c/i.test(e.id);
  const l = s ? { x: 166, y: 238 } : { x: 113, y: 160 },
    u = s ? { x: -53, y: -78 } : { x: 0, y: 0 };
  return c.jsx("svg", {
    width: t || 300,
    viewBox: "0 0 441 403",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    children: c.jsxs("g", {
      id: "XBox",
      children: [
        c.jsx("g", {
          transform: "translate(0, 20)",
          children: c.jsx("path", {
            id: "Outline",
            d: "M220.5 92.0001C200.5 92.0001 154 92.0001 128 92.0001C95.5 92.0001 66.5 109.5 55 137.5C43.5 165.5 4 271.1 4 317.5C4 363.9 17.5 378.5 49.5 378.5C81.5 378.5 105 294.5 150 294.5C195 294.5 220.5 294.5 220.5 294.5C220.5 294.5 245.5 294.5 290.5 294.5C335.5 294.5 359 378.5 391 378.5C423 378.5 436.5 363.9 436.5 317.5C436.5 271.1 397 165.5 385.5 137.5C374 109.5 345 92.0001 312.5 92.0001C286.5 92.0001 240 92.0001 220.5 92.0001Z",
            fill: r,
            fillOpacity: 0.3,
          }),
        }),
        c.jsx("path", {
          id: "Outline",
          d: "M220.5 92.0001C200.5 92.0001 154 92.0001 128 92.0001C95.5 92.0001 66.5 109.5 55 137.5C43.5 165.5 4 271.1 4 317.5C4 363.9 17.5 378.5 49.5 378.5C81.5 378.5 105 294.5 150 294.5C195 294.5 220.5 294.5 220.5 294.5C220.5 294.5 245.5 294.5 290.5 294.5C335.5 294.5 359 378.5 391 378.5C423 378.5 436.5 363.9 436.5 317.5C436.5 271.1 397 165.5 385.5 137.5C374 109.5 345 92.0001 312.5 92.0001C286.5 92.0001 240 92.0001 220.5 92.0001Z",
          stroke: r,
          strokeWidth: a,
          strokeOpacity: 1,
          fill: fe.controllerBodyFill,
        }),
        c.jsx("circle", {
          id: "LStickOutline",
          cx: "113",
          cy: "160",
          r: "37.5",
          stroke: r,
          strokeOpacity: 1,
          strokeWidth: a,
        }),
        c.jsx("circle", {
          id: "RStickOutline",
          cx: "278",
          cy: "238",
          r: "37.5",
          stroke: r,
          strokeOpacity: 1,
          strokeWidth: a,
        }),
        c.jsx("circle", {
          id: "DOutline",
          cx: "166",
          cy: "238",
          r: "37.5",
          stroke: r,
          strokeOpacity: 1,
          strokeWidth: a,
        }),
        c.jsx("circle", {
          id: "LeftStick",
          cx: l.x + e.axes[0] * 12,
          cy: l.y + e.axes[1] * 12,
          r: "28",
          fill: "rgba(0,0,0," + i + ")",
          stroke: "rgba(0,0,0," + (i * 0 + 1) + ")",
          strokeWidth: a,
        }),
        c.jsx("circle", {
          id: "LStickDot",
          cx: l.x + e.axes[0] * 12,
          cy: l.y + e.axes[1] * 12,
          r: 20,
          fill: "rgba(0,0,0," + re(e.buttons[10]) + ")",
          stroke: "rgba(255,255,255," + re(e.buttons[10]) + ")",
          strokeWidth: a,
        }),
        c.jsx("circle", {
          id: "BOutline",
          cx: "329",
          cy: "160",
          r: "37.5",
          stroke: r,
          strokeOpacity: 1,
          strokeWidth: a,
        }),
        c.jsx("circle", {
          id: "RightStick",
          cx: 278 + e.axes[2] * 12,
          cy: 238 + e.axes[3] * 12,
          r: "28",
          fill: "rgba(0,0,0," + o + ")",
          stroke: "rgba(0,0,0," + (o * 0 + 1) + ")",
          strokeWidth: a,
        }),
        c.jsx("circle", {
          id: "RStickDot",
          cx: 278 + e.axes[2] * 12,
          cy: 238 + e.axes[3] * 12,
          r: 20,
          fill: "rgba(0,0,0," + re(e.buttons[11]) + ")",
          stroke: "rgba(255,255,255," + re(e.buttons[11]) + ")",
          strokeWidth: a,
        }),
        c.jsxs("g", {
          transform: `translate(${u.x}, ${u.y})`,
          children: [
            c.jsxs("g", {
              id: "DUp",
              children: [
                c.jsx("mask", {
                  id: "path-8-inside-1",
                  fill: "white",
                  children: c.jsx("path", {
                    d: "M177.669 222.335C180.793 219.21 180.816 213.997 176.868 212.014C176.327 211.743 175.776 211.491 175.215 211.258C172.182 210.002 168.931 209.355 165.648 209.355C162.365 209.355 159.114 210.002 156.081 211.258C155.521 211.491 154.969 211.743 154.429 212.014C150.48 213.997 150.503 219.21 153.627 222.335L159.991 228.698C163.116 231.823 168.181 231.823 171.305 228.698L177.669 222.335Z",
                  }),
                }),
                c.jsx("path", {
                  d: "M177.669 222.335C180.793 219.21 180.816 213.997 176.868 212.014C176.327 211.743 175.776 211.491 175.215 211.258C172.182 210.002 168.931 209.355 165.648 209.355C162.365 209.355 159.114 210.002 156.081 211.258C155.521 211.491 154.969 211.743 154.429 212.014C150.48 213.997 150.503 219.21 153.627 222.335L159.991 228.698C163.116 231.823 168.181 231.823 171.305 228.698L177.669 222.335Z",
                  fill: "rgba(0,0,0," + re(e.buttons[12]) + ")",
                  stroke: "rgba(0,0,0," + (re(e.buttons[12]) * 0 + 1) + ")",
                  strokeWidth: a * 2,
                  mask: "url(#path-8-inside-1)",
                }),
              ],
            }),
            c.jsxs("g", {
              id: "DRight",
              children: [
                c.jsx("mask", {
                  id: "path-9-inside-2",
                  fill: "white",
                  children: c.jsx("path", {
                    d: "M181.447 249.669C184.571 252.793 189.785 252.816 191.768 248.868C192.039 248.327 192.291 247.776 192.523 247.215C193.78 244.182 194.426 240.931 194.426 237.648C194.426 234.365 193.78 231.114 192.523 228.081C192.291 227.521 192.039 226.969 191.768 226.429C189.785 222.48 184.571 222.503 181.447 225.627L175.083 231.991C171.959 235.116 171.959 240.181 175.083 243.305L181.447 249.669Z",
                  }),
                }),
                c.jsx("path", {
                  d: "M181.447 249.669C184.571 252.793 189.785 252.816 191.768 248.868C192.039 248.327 192.291 247.776 192.523 247.215C193.78 244.182 194.426 240.931 194.426 237.648C194.426 234.365 193.78 231.114 192.523 228.081C192.291 227.521 192.039 226.969 191.768 226.429C189.785 222.48 184.571 222.503 181.447 225.627L175.083 231.991C171.959 235.116 171.959 240.181 175.083 243.305L181.447 249.669Z",
                  fill: "rgba(0,0,0," + re(e.buttons[15]) + ")",
                  stroke: "rgba(0,0,0," + (re(e.buttons[15]) * 0 + 1) + ")",
                  strokeWidth: a * 2,
                  mask: "url(#path-9-inside-2)",
                }),
              ],
            }),
            c.jsxs("g", {
              id: "DDown",
              children: [
                c.jsx("mask", {
                  id: "path-10-inside-3",
                  fill: "white",
                  children: c.jsx("path", {
                    d: "M154.113 253.447C150.989 256.571 150.966 261.785 154.914 263.767C155.455 264.039 156.006 264.291 156.566 264.523C159.6 265.78 162.85 266.426 166.134 266.426C169.417 266.426 172.667 265.78 175.701 264.523C176.261 264.291 176.812 264.039 177.353 263.767C181.301 261.785 181.279 256.571 178.154 253.447L171.79 247.083C168.666 243.959 163.601 243.959 160.477 247.083L154.113 253.447Z",
                  }),
                }),
                c.jsx("path", {
                  d: "M154.113 253.447C150.989 256.571 150.966 261.785 154.914 263.767C155.455 264.039 156.006 264.291 156.566 264.523C159.6 265.78 162.85 266.426 166.134 266.426C169.417 266.426 172.667 265.78 175.701 264.523C176.261 264.291 176.812 264.039 177.353 263.767C181.301 261.785 181.279 256.571 178.154 253.447L171.79 247.083C168.666 243.959 163.601 243.959 160.477 247.083L154.113 253.447Z",
                  fill: "rgba(0,0,0," + re(e.buttons[13]) + ")",
                  stroke: "rgba(0,0,0," + (re(e.buttons[13]) * 0 + 1) + ")",
                  strokeWidth: a * 2,
                  mask: "url(#path-10-inside-3)",
                }),
              ],
            }),
            c.jsxs("g", {
              id: "DLeft",
              children: [
                c.jsx("mask", {
                  id: "path-11-inside-4",
                  fill: "white",
                  children: c.jsx("path", {
                    d: "M150.335 226.113C147.21 222.989 141.997 222.966 140.014 226.914C139.743 227.455 139.491 228.006 139.258 228.566C138.002 231.6 137.355 234.85 137.355 238.134C137.355 241.417 138.002 244.667 139.258 247.701C139.491 248.261 139.743 248.812 140.014 249.353C141.997 253.301 147.21 253.279 150.335 250.154L156.698 243.79C159.823 240.666 159.823 235.601 156.698 232.477L150.335 226.113Z",
                  }),
                }),
                c.jsx("path", {
                  d: "M150.335 226.113C147.21 222.989 141.997 222.966 140.014 226.914C139.743 227.455 139.491 228.006 139.258 228.566C138.002 231.6 137.355 234.85 137.355 238.134C137.355 241.417 138.002 244.667 139.258 247.701C139.491 248.261 139.743 248.812 140.014 249.353C141.997 253.301 147.21 253.279 150.335 250.154L156.698 243.79C159.823 240.666 159.823 235.601 156.698 232.477L150.335 226.113Z",
                  fill: "rgba(0,0,0," + re(e.buttons[14]) + ")",
                  stroke: "rgba(0,0,0," + (re(e.buttons[14]) * 0 + 1) + ")",
                  strokeWidth: a * 2,
                  mask: "url(#path-11-inside-4)",
                }),
              ],
            }),
          ],
        }),
        c.jsxs("g", {
          id: "BTop",
          children: [
            c.jsx("mask", {
              id: "path-13-inside-5",
              fill: "white",
              children: c.jsx("path", {
                d: "M340.669 144.335C343.793 141.21 343.816 135.997 339.868 134.014C339.327 133.743 338.776 133.491 338.215 133.258C335.182 132.002 331.931 131.355 328.648 131.355C325.365 131.355 322.114 132.002 319.081 133.258C318.521 133.491 317.969 133.743 317.429 134.014C313.48 135.997 313.503 141.21 316.627 144.335L322.991 150.698C326.116 153.823 331.181 153.823 334.305 150.698L340.669 144.335Z",
              }),
            }),
            c.jsx("path", {
              d: "M340.669 144.335C343.793 141.21 343.816 135.997 339.868 134.014C339.327 133.743 338.776 133.491 338.215 133.258C335.182 132.002 331.931 131.355 328.648 131.355C325.365 131.355 322.114 132.002 319.081 133.258C318.521 133.491 317.969 133.743 317.429 134.014C313.48 135.997 313.503 141.21 316.627 144.335L322.991 150.698C326.116 153.823 331.181 153.823 334.305 150.698L340.669 144.335Z",
              fill: "rgba(0,0,0," + re(e.buttons[3]) + ")",
              stroke: "rgba(0,0,0," + (re(e.buttons[3]) * 0 + 1) + ")",
              strokeWidth: a * 2,
              mask: "url(#path-13-inside-5)",
            }),
          ],
        }),
        c.jsxs("g", {
          id: "BRight",
          children: [
            c.jsx("mask", {
              id: "path-14-inside-6",
              fill: "white",
              children: c.jsx("path", {
                d: "M344.447 171.669C347.571 174.793 352.785 174.816 354.768 170.868C355.039 170.327 355.291 169.776 355.523 169.215C356.78 166.182 357.426 162.931 357.426 159.648C357.426 156.365 356.78 153.114 355.523 150.081C355.291 149.521 355.039 148.969 354.768 148.429C352.785 144.48 347.571 144.503 344.447 147.627L338.083 153.991C334.959 157.116 334.959 162.181 338.083 165.305L344.447 171.669Z",
              }),
            }),
            c.jsx("path", {
              d: "M344.447 171.669C347.571 174.793 352.785 174.816 354.768 170.868C355.039 170.327 355.291 169.776 355.523 169.215C356.78 166.182 357.426 162.931 357.426 159.648C357.426 156.365 356.78 153.114 355.523 150.081C355.291 149.521 355.039 148.969 354.768 148.429C352.785 144.48 347.571 144.503 344.447 147.627L338.083 153.991C334.959 157.116 334.959 162.181 338.083 165.305L344.447 171.669Z",
              fill: "rgba(0,0,0," + re(e.buttons[1]) + ")",
              stroke: "rgba(0,0,0," + (re(e.buttons[1]) * 0 + 1) + ")",
              strokeWidth: a * 2,
              mask: "url(#path-14-inside-6)",
            }),
          ],
        }),
        c.jsxs("g", {
          id: "BBottom",
          children: [
            c.jsx("mask", {
              id: "path-15-inside-7",
              fill: "white",
              children: c.jsx("path", {
                d: "M317.113 175.447C313.989 178.571 313.966 183.785 317.914 185.767C318.455 186.039 319.006 186.291 319.566 186.523C322.6 187.78 325.85 188.426 329.134 188.426C332.417 188.426 335.667 187.78 338.701 186.523C339.261 186.291 339.812 186.039 340.353 185.767C344.301 183.785 344.279 178.571 341.154 175.447L334.79 169.083C331.666 165.959 326.601 165.959 323.477 169.083L317.113 175.447Z",
              }),
            }),
            c.jsx("path", {
              d: "M317.113 175.447C313.989 178.571 313.966 183.785 317.914 185.767C318.455 186.039 319.006 186.291 319.566 186.523C322.6 187.78 325.85 188.426 329.134 188.426C332.417 188.426 335.667 187.78 338.701 186.523C339.261 186.291 339.812 186.039 340.353 185.767C344.301 183.785 344.279 178.571 341.154 175.447L334.79 169.083C331.666 165.959 326.601 165.959 323.477 169.083L317.113 175.447Z",
              fill: "rgba(0,0,0," + re(e.buttons[0]) + ")",
              stroke: "rgba(0,0,0," + (re(e.buttons[0]) * 0 + 1) + ")",
              strokeWidth: a * 2,
              mask: "url(#path-15-inside-7)",
            }),
          ],
        }),
        c.jsxs("g", {
          id: "BLeft",
          children: [
            c.jsx("mask", {
              id: "path-16-inside-8",
              fill: "white",
              children: c.jsx("path", {
                d: "M313.335 148.113C310.21 144.989 304.997 144.966 303.014 148.914C302.743 149.455 302.491 150.006 302.258 150.566C301.002 153.6 300.355 156.851 300.355 160.134C300.355 163.417 301.002 166.668 302.258 169.701C302.491 170.261 302.743 170.812 303.014 171.353C304.997 175.301 310.21 175.279 313.335 172.154L319.698 165.79C322.823 162.666 322.823 157.601 319.698 154.477L313.335 148.113Z",
              }),
            }),
            c.jsx("path", {
              d: "M313.335 148.113C310.21 144.989 304.997 144.966 303.014 148.914C302.743 149.455 302.491 150.006 302.258 150.566C301.002 153.6 300.355 156.851 300.355 160.134C300.355 163.417 301.002 166.668 302.258 169.701C302.491 170.261 302.743 170.812 303.014 171.353C304.997 175.301 310.21 175.279 313.335 172.154L319.698 165.79C322.823 162.666 322.823 157.601 319.698 154.477L313.335 148.113Z",
              fill: "rgba(0,0,0," + re(e.buttons[2]) + ")",
              stroke: "rgba(0,0,0," + (re(e.buttons[2]) * 0 + 1) + ")",
              strokeWidth: a * 2,
              mask: "url(#path-16-inside-8)",
            }),
          ],
        }),
        c.jsx("g", {
          id: "LMeta",
          children: c.jsx("circle", {
            cx: "185",
            cy: "162",
            r: "10",
            fill: "rgba(0,0,0," + re(e.buttons[8]) + ")",
            stroke: "rgba(0,0,0," + (re(e.buttons[8]) * 0 + 1) + ")",
            strokeWidth: a,
          }),
        }),
        c.jsx("g", {
          id: "RMeta",
          children: c.jsx("circle", {
            cx: "259",
            cy: "162",
            r: "10",
            fill: "rgba(0,0,0," + re(e.buttons[9]) + ")",
            stroke: "rgba(0,0,0," + (re(e.buttons[9]) * 0 + 1) + ")",
            strokeWidth: a,
          }),
        }),
        c.jsx("rect", {
          id: "L1",
          x: "111.5",
          y: "61.5",
          width: "41",
          height: "13",
          rx: "6.5",
          fill: "rgba(0,0,0," + re(e.buttons[4]) + ")",
          stroke: "rgba(0,0,0," + (re(e.buttons[4]) * 0 + 1) + ")",
          strokeWidth: a,
        }),
        c.jsx("rect", {
          id: "R1",
          x: "289.5",
          y: "61.5",
          width: "41",
          height: "13",
          rx: "6.5",
          fill: "rgba(0,0,0," + re(e.buttons[5]) + ")",
          stroke: "rgba(0,0,0," + (re(e.buttons[5]) * 0 + 1) + ")",
          strokeWidth: a,
        }),
        c.jsx("path", {
          id: "L2",
          d: "M152.5 37C152.5 41.1421 149.142 44.5 145 44.5H132C127.858 44.5 124.5 41.1421 124.5 37V16.5C124.5 8.76801 130.768 2.5 138.5 2.5C146.232 2.5 152.5 8.76801 152.5 16.5V37Z",
          fill: "rgba(0,0,0," + re(e.buttons[6]) + ")",
          stroke: "rgba(0,0,0," + (re(e.buttons[6]) * 0 + 1) + ")",
          strokeWidth: a,
        }),
        c.jsx("path", {
          id: "R2",
          d: "M317.5 37C317.5 41.1421 314.142 44.5 310 44.5H297C292.858 44.5 289.5 41.1421 289.5 37V16.5C289.5 8.76801 295.768 2.5 303.5 2.5C311.232 2.5 317.5 8.76801 317.5 16.5V37Z",
          fill: "rgba(0,0,0," + re(e.buttons[7]) + ")",
          stroke: "rgba(0,0,0," + (re(e.buttons[7]) * 0 + 1) + ")",
          strokeWidth: a,
        }),
        c.jsx("line", {
          x1: "30",
          y1: "210",
          x2: "130",
          y2: "300",
          strokeWidth: a,
          stroke: r,
          opacity: 0.3,
        }),
        c.jsx("line", {
          x1: 411,
          y1: "210",
          x2: 311,
          y2: "300",
          strokeWidth: a,
          stroke: r,
          opacity: 0.3,
        }),
      ],
    }),
  });
}
function buttonValue(buttonLike) {
  if (buttonLike === undefined || buttonLike === null) {
    return 0;
  }
  if (typeof buttonLike === "number") {
    return buttonLike;
  }
  if (typeof buttonLike.value === "number") {
    return buttonLike.value;
  }
  return 0;
}

function normalizeGamepad(gamepad) {
  const axes = [0, 0, 0, 0];
  if (
    gamepad !== null &&
    gamepad !== undefined &&
    Array.isArray(gamepad.axes)
  ) {
    for (let i = 0; i < Math.min(4, gamepad.axes.length); i += 1) {
      const value = Number(gamepad.axes[i]);
      axes[i] = Number.isFinite(value) ? value : 0;
    }
  }

  const buttons = [];
  const sourceButtons =
    gamepad !== null && gamepad !== undefined && Array.isArray(gamepad.buttons)
      ? gamepad.buttons
      : [];
  for (let i = 0; i < 17; i += 1) {
    buttons.push(buttonValue(sourceButtons[i]));
  }

  return {
    id:
      gamepad !== null &&
        gamepad !== undefined &&
        typeof gamepad.id === "string"
        ? gamepad.id
        : "No Gamepad",
    index:
      gamepad !== null &&
        gamepad !== undefined &&
        typeof gamepad.index === "number"
        ? gamepad.index
        : 0,
    axes,
    buttons,
  };
}

function pickPrimaryGamepad() {
  if (!navigator.getGamepads) {
    return null;
  }

  const pads = navigator.getGamepads();
  if (!pads) {
    return null;
  }

  for (const pad of pads) {
    if (pad !== null && pad !== undefined && pad.connected) {
      return pad;
    }
  }

  return null;
}

const root = document.getElementById("gamepad-root");
let lastSignature = "";

function signatureOf(gamepad) {
  const axisPart = gamepad.axes.map((axis) => axis.toFixed(4)).join("|");
  const buttonPart = gamepad.buttons
    .map((button) => button.toFixed(4))
    .join("|");
  return `${gamepad.id}|${gamepad.index}|${axisPart}|${buttonPart}`;
}

function renderGamepad(gamepad) {
  currentGamepad = gamepad;
  if (root === null) {
    return;
  }
  const svg = A5({ gamepad, width: 441 });
  root.replaceChildren(svg);
}

function frame() {
  const normalized = normalizeGamepad(pickPrimaryGamepad());
  const signature = signatureOf(normalized);
  if (signature !== lastSignature) {
    lastSignature = signature;
    renderGamepad(normalized);
  }
  requestAnimationFrame(frame);
}

const initialSystemTheme =
  window.matchMedia &&
  window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
applyTheme(initialSystemTheme);

if (window.matchMedia) {
  const darkThemeMedia = window.matchMedia("(prefers-color-scheme: dark)");
  const onThemeChanged = (event) => {
    applyTheme(event.matches ? "dark" : "light");
  };
  if (typeof darkThemeMedia.addEventListener === "function") {
    darkThemeMedia.addEventListener("change", onThemeChanged);
  } else if (typeof darkThemeMedia.addListener === "function") {
    darkThemeMedia.addListener(onThemeChanged);
  }
}

renderGamepad(normalizeGamepad(null));
frame();

window.addEventListener("gamepadconnected", () => {
  lastSignature = "";
});
window.addEventListener("gamepaddisconnected", () => {
  lastSignature = "";
});

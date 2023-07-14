import * as ClassUtils from "./class.js";
import { CY_CLASSES } from "../../constants.js";

/**
 * Returns a generator that generates distinct hues on [0, 360)
 * @returns hue
 */
export function* genDistinctHues(base = 2) {
  yield 0;
  let increment = 360;
  while (true) {
    increment /= base;
    for (let i = increment; i < 360; i += increment * base) {
      yield i;
    }
  }
}

/**
 * Generate a Cytoscape stylesheet
 * @param {Object} keyedGroup elements format for Cytoscape
 * @param {string} outputPath (optional) path to output json
 * @returns
 */
function generateStylesheet(keyedGroup) {
  let stylesheet = [
    {
      selector: "node[label]",
      style: {
        label: "data(label)",
      },
    },
    {
      selector: "edge[label]",
      style: {
        "label": "data(label)",
        "text-rotation": "autorotate",
        "text-margin-x": 0,
        "text-margin-y": 0,
      },
    },
    {
      selector: "node:childless",
      style: {
        "label": "data(label)",
        "text-halign": "center",
        "text-valign": "center",
        "width": 160,
        "height": 70,
        "text-max-width": "140",
        "text-wrap": "wrap",
        "color": "white",
        "text-outline-color": "grey",
        "text-outline-width": 2,
      },
    },
    {
      selector: "node.node:childless",
      style: {
        "background-color": "#000000",
      },
    },
    {
      selector: "node.edge",
      style: {
        "shape": "rectangle",
        "background-color": "#5A5A5A",
      },
    },
    {
      selector: "edge",
      style: {
        "curve-style": "bezier",
        "target-arrow-shape": "vee",
        "arrow-scale": 3,
        "color": "white",
        "text-outline-color": "grey",
        "text-outline-width": 1,
      },
    },
  ];
  // styles appended to end of styleSheet after it being sorted by selector specitivity
  const appendedStyles = [
    {
      selector: ".highlight[^properties.composite]",
      style: {
        "background-color": "red",
      },
    },
    {
      selector: "edge.unhighlighted",
      style: {
        opacity: 0.1,
      },
    },
    {
      selector: ":selected",
      style: {
        "background-color": "blue",
      },
    },
    {
      selector: ".highlight:selected[^properties.composite]",
      style: {
        "background-color": "#74008B", // nicer purple
      },
    },
  ];

  const allClasses = new Set(
    keyedGroup.nodes
      .filter((elem) => elem.classes.includes("edge"))
      .map((edge) =>
        ClassUtils.getHighestOrderClass(
          ClassUtils.filterClassesByPrefix(edge.classes, [
            `${CY_CLASSES.CONTROL}-`,
            `${CY_CLASSES.COORDINATES}-`,
          ])
        )
      )
  );
  const colourGenerator = genDistinctHues();

  for (const classs of allClasses) {
    const hslColour = `hsl(${colourGenerator.next().value},90%,50%)`;
    stylesheet.push({
      selector: "." + classs,
      style: {
        "line-color": hslColour,
        "target-arrow-color": hslColour,
      },
    });
  }

  // ensures most-specific style is appended to stylesheet last (so it is chosen)
  stylesheet.sort((a, b) => a.selector - b.selector);
  stylesheet = [...stylesheet, ...appendedStyles];

  return stylesheet;
}

export { generateStylesheet };

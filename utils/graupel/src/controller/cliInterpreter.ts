import { CurrentInstance, Instances } from "../model/instanceManager";
import * as Manipulator from "./manipulator";
import { Aliases } from "./manipulatorAliases.js";

let cyFunctions = undefined;
let userFunctions = undefined;
let aliasFunctions = undefined;
let cyAndUserFuncs = undefined;
let commands = undefined;

/**
 * Creates the list of commands availible from Cytoscape, UserFuncs and Aliases
 * @param Mounted.cy
 * @returns object containing {commandName: Function} pairs
 */

function createCommands() {
  function _createCommands() {
    console.log("Setting up CLI");
    cyFunctions = {};
    for (const key in CurrentInstance.cy) {
      if (typeof CurrentInstance.cy[key] === "function") {
        cyFunctions[key] = (...__rocRollerArgs) =>
          CurrentInstance.cy[key](...__rocRollerArgs);
        // bind doesn't work as it needs to be bound to current instance, i.e.
        // commands[key] = Mounted.cy[key].bind(Mounted.cy);
      }
    }

    // UserFuncs
    userFunctions = {};
    for (const key in Manipulator) {
      if (key in cyFunctions) {
        console.log(
          "Overwriting Cytoscape function `" +
          key +
          "` with UserFuncs function of the same name"
        );
      }
      userFunctions[key] = Manipulator[key];
    }

    // Aliases of Cytoscape functions or UserFuncs
    aliasFunctions = {};
    for (const funcName of Object.keys(Aliases)) {
      for (const aliasName of Aliases[funcName]) {
        if (!(funcName in { ...cyFunctions, ...userFunctions })) {
          console.log(
            `Alias ${aliasName} for non-existant function ${funcName}, alias not added. `
          )
          continue;
        }
        if (aliasName in { ...cyFunctions, ...userFunctions }) {
          console.log(
            "Overwriting `" +
            aliasName +
            "` with an alias for `" +
            funcName +
            "`"
          );
        }
        aliasFunctions[aliasName] = userFunctions[funcName];
      }
    }

    return [
      { ...cyFunctions, ...userFunctions },
      { ...cyFunctions, ...userFunctions, ...aliasFunctions },
    ];
  }

  if (
    typeof commands === "undefined" ||
    typeof cyAndUserFuncs === "undefined"
  ) {
    [cyAndUserFuncs, commands] = _createCommands();
  }
}

CurrentInstance.cy.ready(createCommands);

/**
 * Get a stringified of array of function names and their parameters
 * @param Mounted.cy
 * @returns
 */
export function getPrettifiedFunctionList() {
  function getParamNames(key, func) {
    const STRIP_COMMENTS =
      /(\/\/.*$)|(\/\*[\s\S]*?\*\/)|(\s*=[^,\)]*(('(?:\\'|[^'\r\n])*')|("(?:\\"|[^"\r\n])*"))|(\s*=[^,\)]*))/gm;
    const ARGUMENT_NAMES = /([^\s,]+)/g;

    const functionString = func.toString().replace(STRIP_COMMENTS, "");

    let result = functionString
      .slice(functionString.indexOf("(") + 1, functionString.indexOf(")"))
      .match(ARGUMENT_NAMES);

    if (key in cyFunctions) {
      result = getParamNames(undefined, CurrentInstance.cy[key]);
    }

    if (result === null) result = [];

    return result;
  }

  createCommands();
  let output = {};
  for (const key in cyAndUserFuncs) {
    output[key] = getParamNames(key, cyAndUserFuncs[key]);
  }
  return output;
}

/**
 * Parses a command
 * @param Mounted.cy
 * @param input string
 * @returns the return of the command
 */
export function parse(input: string) {
  function runCode(code: string) {
    try {
      console.log("Running: ", code); // cannot directly eval due to lack of global-scope cy object
      return Instances.doEval(code);
    } catch (e) {
      return undefined;
    }
  }

  if (input.startsWith("`")) {
    return runCode(input.substring(1));
  } else if (input.startsWith(".")) {
    return runCode("Mounted.cy" + input);
  }

  createCommands();

  const args = input.split(" ");
  let funcArgs: any = undefined;

  try {
    funcArgs = eval(`[${args.slice(1).toString()}]`); // for non-string literal arguments
  } catch (e) {
    funcArgs = args.slice(1);
  }

  for (const key in commands) {
    if (args[0] === key) {
      console.log("Calling", key, " with ", funcArgs);
      return commands[key](...funcArgs);
    }
    if (args[0].toLowerCase() === key.toLowerCase()) {
      console.log("Calling", key, " with ", funcArgs);
      return commands[key](...funcArgs);
    }
  }

  console.log("Parsing: ", [args[0], ...funcArgs]);
  console.log("Invalid command, options are: ", getPrettifiedFunctionList());
}

export function getAllFunctions() {
  return [cyFunctions, userFunctions, aliasFunctions];
}

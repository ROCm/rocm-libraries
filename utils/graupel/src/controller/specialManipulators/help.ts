module Help {
  const prompts = [
    "Welcome to the help series. Press the 'Help' button on the left-hand-side to continue (you might want to spam it until the end)",

    "Holding <control> and pressing <space> will highlight the CLI. Then press <enter> to search!",

    "Try running the command `idselect coord1`. Hold right click on the node to see some actions!",

    "You can use your mouse to move around, and hold <shift> then click and drag to make boxes. ",

    "I recommend running `npm run docs` to generate docs (hacky method to-be-fixed). ",

    "For more commands, type `man` and to get help on a specific one, use `man <name>`, e.g. `man newInstance`. ",

    "If you want to load a new graph, use `new path/to/kernel.s` (or .yaml), e.g. `new kernels/sample.s`. This is an alias for `newInstance`",

    "You can also append `#path/to/kernel.s` to the URL then reload the page; the bash script `./url.sh <file-path>` will also echo the URL (if you want auto-complete on paths)", 

    "The end :)",
  ];
  let promptCounter = 0;
  export function f() {
    promptCounter = Math.min(promptCounter + 1, prompts.length - 1);
    return prompts[promptCounter];
  }
  export function b() {
    promptCounter = Math.max(promptCounter - 1, 0);
    return prompts[promptCounter];
  }
}

export default Help;

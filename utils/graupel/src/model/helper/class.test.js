import * as ClassUtils from "./class.js";

test("getOrderOfClass", () => {
  const classes = [
    "one-two-three",
    "one-two",
    "one",
    "one-eleven-twelve",
    "four-five",
    "four-five-six-seven",
  ];
  const actual = classes.map(ClassUtils.getOrderOfClass);
  expect(actual).toStrictEqual([2, 1, 0, 2, 1, 3]);
});

test("getHighestOrderClass", () => {
  const classes = [
    "one-two-three",
    "one-two",
    "one",
    "one-eleven-twelve",
    "four-five",
    "four-five-six-seven",
  ];
  const actual = ClassUtils.getHighestOrderClass(classes);
  expect(actual).toBe("four-five-six-seven");
});

test("filterClassesByPrefix", () => {
  const classes = [
    "one-two-three",
    "one-two",
    "one",
    "one-eleven-twelve",
    "four-five",
    "four-five-six-seven",
  ];
  const actual = ClassUtils.filterClassesByPrefix(classes, ["four-"]);
  expect(actual).toStrictEqual(["four-five", "four-five-six-seven"]);
});

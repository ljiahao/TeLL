# Tree-sitter Language binding

We use `tree-sitter` to transform java source into Abstract Syntax Tree. In order to utilize python, you need binding specific languages with py-tree-sitter. 

Here, we provide one library file `my_languages.so` binding `java`, `c` used in our experiments. You can use it directly. If you want to generate the library file by yourself. You can use our build script or visit tree-sitter github repo for detailed solutions via this [link](https://github.com/tree-sitter/py-tree-sitter).
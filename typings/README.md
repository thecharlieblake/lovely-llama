# Type stubs

Pyright isn't happy with jax out-of-the-box, largely because it imports numpy typing without using proper generics (e.g. `np.number` instead of `np.number[Any]`).

The best way of fixing this without turning off key type-checking is as follows:
1. Identify the lowest-level `.py` or `.pyi` file which causes the error
2. Copy the file into the `typings/` directory with the same file-structure as before
3. Modify the file to fix the typing issue (making sure not to reformat, as this makes diffing painful)
4. Pyright will pick this file up and use it for typing instead of the orignal - problem solved!

To understand the changes made, developers are encouraged to diff against the equivalent jax files (these are easy to find as the file-structure is the same here as in the jax repo, by necessity).
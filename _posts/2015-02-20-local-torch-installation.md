---
layout: post
title: Local Torch installation
author: gustav
---

This post describes how to do a local Torch7 installation while ignoring a
potentially conflicting global installation in `/usr/local/share`.

Doing a local Torch7 installation is easily done using
[torch/distro](https://github.com/torch/distro). However, when running
`install.sh`, I ran into the following error:

```
/usr/larsson/torch/install/bin/luajit: /tmp/luarocks_cutorch-scm-1-5301/cutorch/TensorMath.lua:184: attempt to call method 'registerDefaultArgument' (a nil value)
stack traceback:
        /tmp/luarocks_cutorch-scm-1-5301/cutorch/TensorMath.lua:184: in main chunk
        [C]: at 0x00405330
make[2]: *** [TensorMath.c] Error 1
make[1]: *** [CMakeFiles/cutorch.dir/all] Error 2
make: *** [all] Error 2
```

This issue is documented [here](https://github.com/torch/cutorch/issues/106)
and the solution is to remove the global installation in `/usr/local/share`.
This was not an option for me. This is what I did.

I cloned `torch/distro` as you would, let us say to `~/torch`:

```
git clone git@github.com:torch/distro.git ~/torch --recursive
```

I went into `~/torch` and ran `install.sh`, which failed. For me, I still got
Torch installed even though some packages failed. Check that this is the case by
running `which th` and `which luarocks` - it should point to `~/torch`. If this
is the case, run `th` and type in:

```
> print(package.path)
> print(package.cpath)
```

Copy these strings to your `LUA_PATH` and `LUA_CPATH`, respectively. Leave out
any references to `/usr/local/share`! This might look something like this in
your `~/.bashrc`:

```
export TORCH_DIR=$HOME/torch
export LUA_PATH="$TORCH_DIR/install/share/lua/5.1/?.lua;$TORCH_DIR/install/share/lua/5.1/?/init.lua;$TORCH_DIR/install/share/luajit-2.1.0-alpha/?.lua"
export LUA_CPATH="$TORCH_DIR/install/lib/lua/5.1/?.so"
```

Note that you have to quote the strings since their usage of `;` as a delimiter
does not play well with bash. Once saved, refresh your shell by running `source ~/.bashrc` and try
installing the packages that failed. I did

```
luarocks install cutorch
luarocks install cunn
```

This time around it worked and I was good to go.

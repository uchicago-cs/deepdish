---
layout: post
author: gustav
title: GNU Parallel
---

I was reading the [ImageNet
tutorial](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html) for
[Caffe](http://caffe.berkeleyvision.org/) (a deep learning framework), in which
they need to resize a large number of images. It struck me that they might not
be aware of [GNU Parallel](http://www.gnu.org/software/parallel/), since it
is a great tool for this task. I recommend it to any data scientist out there
since it is so simple to use and like many other GNU tools, with good chance
already installed on your computer (if not, `apt-get install moreutils` on
Debian).

In the writeup, it says that the author used his own MapReduce framework to do
it, but it can also be done sequentially as:

```sh
for name in *.jpeg; do
    convert -resize 256x256\! $name $name
done
```

Instead of this sequential approach, you can run it in parallel with even less
typing:

```sh
parallel convert -resize 256x256\! {} {} ::: *.jpeg
```

GNU Parallel will insert each filename at `{}` to form a command. Multiple
commands will execute concurrently if you have a multicore computer.

If you have ever been tempted to do this kind of parallelization by adding `&`
at the end of each command in the for loop, then Parallel is definetely for
you. Adding `&` introduces two problems that Parallel solves: (1) you don't
know when all of them are done and there is no easy way to _join_ them, and (2)
it will start a process for each command all at once, while Parallel will
schedule your tasks and execute only as many in parallel as your computer
can handle.

## Basics

Parallel can also take input from the pipe, in which case it is similar to xargs:

```
ls *.jpeg | parallel mv {} {.}-old.jpeg
```

This command inserts `-old` into the filenames of all the JPEG files in the
directory. The `{.}` is similar to `{}`, except it removes the extension. There
are many replacement strings like this:

```sh
parallel convert -resize 256x256\! {} resized/{/} ::: images/*.jpeg
```

This resizes all the JPEG files inside the folder `images` and places the
output in the folder `resized`. The replacement string `{/}` extracts the
filename and is thus similar to the command `basename`. For this example we
went back to the `:::` style input, which in many cases is preferable. For
instance, it can be used several times to form a product of the input:

```bash
parallel "echo {1}: {2}" ::: A B C D ::: {1..8}
```

Note how we now used `{1}` and `{2}` to refer to the input. We also quoted the
command, which is optional and might make things clearer (if you want to use
pipes inside your command, it is required). Using multiple inputs is great for
doing grid searches of parameters. However, let's say we don't want to do all
combinations in the product and instead want to specify each pair of input
manually. First create a file with the input and name it `input.txt`:

```
A 10
B 20
C 10
```

Now, use `--colsep` to specify the delimiter:

```sh
parallel --colsep=' ' "echo {1}: {2}" < input.txt
```

If you did this to test a variety of parameters, you might find it easier to
create a file, `commands.sh`,  with all the commands written out:

```sh
./experiment 10.0 1.5 > exp1.txt
./experiment 20.0 1.5 --extra-param 3.0 > exp2.txt
```
Now run them in parallel by:

```
parallel < commands.sh          # OR
parallel :::: commands.sh
```

The latter is a newer syntax (note that it has *four* semicolons), which again I
prefer since it can be stringed together multiple times and you can freely mix
`:::` and `::::`.


## Multiple computers using SSH

Parallel can also be used to parallelize between multiple computers. Let's say
you have SSH access to the hostnames or SSH aliases `node1` and `node2` without
prompting for password. Now you can tell Parallel to distribute the job across
both nodes using the `-S` option:

```sh
parallel -S node1,node2 -j8 convert -resize 256x256\! {} {} ::: *.jpeg
```

You can refer to the local computer as `:` (e.g. do `-S :,node1,node2` to
include the current computer). I also added `-j8` to specify that I want each
node to run 8 jobs concurrently. You can try leaving this out, but Parallel
could have a hard time automatically determining how many jobs to use for each
node.

We assumed in this example that the files existed on the other nodes (for
instance through NSF). However, Parallel can also transfer the files to the
worker nodes and transfer the results back by adding `--trc {}`.

## More information

For more information I recommend:

* [GNU Parallel Tutorial](http://www.gnu.org/software/parallel/parallel_tutorial.html) - Very readable with lots of information
* [GNU Parallel Videos](https://www.youtube.com/playlist?list=PL284C9FF2488BC6D1) - Screencasts by the author of Parallel
* [Parallel Batch Job Submission](http://docs.rcc.uchicago.edu/software/scheduler/parallel/README.html) - How to use Parallel on a SLURM cluster

<!---
## Multiple computers using a cluster

This will depenend a bit on your cluster and its scheduler. However, as an
example, we run things on University of Chicago's [RCC
cluster](http://rcc.uchicago.edu/) which uses the scheduler SLURM. Batch jobs
are submitted using `sbatch`, but sub-jobs can be submitted inside your batch
job using `srun`. So, in order to use Parallel across the cluster, we can submit
a batch job that looks like this:

```sh
#SBATCH --ntasks 64
#SBATCH --exclusive

parallel -n500 --delay 0.2 -j4 "srun --exclusive -N1 ./batch-resize.sh" ::: *.jpeg
```

To avoid sending too many `srun`, we have added a small delay and split it up
into batches of 500.  The `-n500` means it will send 500 filenames to each
command, so one call to `srun` will process 500 images. For `srun`, we specify
`-N1` in order to send it to one node only. The `batch-resize.sh` takes any
number of parameters and performs a resize on all of them, so it might look like:

```sh
parallel convert -resize 256x256\! {} {} ::: $@
```

This may not be relevant to your cluster, but the idea could be similar. You
might also be able to do the SSH solution if you know the hostnames of your
worker nodes. Check with your cluster's staff since chances are they know about
GNU Parallel and how to deploy it onto the cluster.
-->

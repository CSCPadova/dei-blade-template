# Using Blade

Use copier to generate a new project from this template.

```bash
pip install copier
copier copy gh:CSCPadova/dei-blade-template my_project
cd my_project
```

Modify it as you want and save the files.

Once done you are ready to use the cluster, be sure to have an account and an ssh terminal.

{: .important }
> You can activate your account by following the instructions at [this page](https://www.dei.unipd.it/account).

## Copy the files to the cluster

In order to be able to execute your code on the cluster, you need to copy the files to the cluster.

```bash
scp -r . <username>@login.dei.unipd.it:path/to/the/project
```

With this command you will copy all the files in the current directory to the specified path on the cluster (if omitted the first `/` the files will be copied at the specified path relative to the home directory).

## Connect to the cluster

Now that everything is set up, you can connect to the cluster using the following command:

```bash
ssh <username>@login.dei.unipd.it
```

You will be asked for your password, insert it and you will be connected to the cluster.

### Install uv

The first time you connect to the cluster you need to install the `uv` command, to do so you can run the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

It will install the `uv` command in your home directory, under the `.local/bin` directory, if you want to use the uv command in the terminal please make sure to add the `.local/bin` directory to your PATH.

### Run the code

Now that you are connected to the cluster you can run your code using the following command:

```bash
make train # or make infer
```

This command will run the training or inference script on the cluster and, if needed, will install the required project's dependencies.

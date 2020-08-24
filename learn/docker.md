# `Docker`

[TOC]

## 一、`Docker` 简介

## 二、`Docker` 安装

## 三、`Docker` 常用命令

### 帮助命令

**`docker version`**

**`docker info`**

**`docker --help`**

### 镜像命令

#### `docker images`

列出本机上所有镜像

**表项说明：**

1. `REPOSITORY`：镜像的仓库源
2. `TAG`：镜像的标签
3. `IMAGE ID`：镜像 `ID`
4. `CREATED`：镜像创建时间
5. `SIZE`：镜像大小

> 同一个仓库源可以有多个 `TAG`，表示这个仓库源的不同版本，使用 `REPOSITORY:TAG` 来定义不同的镜像。
>
> 如果不指定一个仓库的版本标签，例如只使用 `ubuntu`，`docker` 将默认使用 `ubuntu:latest` 镜像

**`OPTIONS`：**

- `-a`：列出本地所有镜像（含中间镜像层）
- `-q`：只显示镜像 `ID`
- `--digests`：显示镜像的摘要信息
- `--no-trunc`：显示完整的镜像信息

**`OPTIONS` 组合：**

- `-qa`：本地所有镜像的 `ID`

#### `docker search`

在 `docker hub` 上搜索镜像

**`OPTIONS:`**

- `--no-trunc`：显示完整的镜像描述
- `-s`：列出收藏数不小于指定值的镜像
- `--automated`：只列出 `automated build` 类型的镜像

#### `docker pull`

从远程下载镜像，如果镜像名没有指定 `TAG`，则默认选择 `latest`，例如 `tensorflow:latest`

#### `docker rmi`

删除镜像，默认删 `latest`，删除多个镜像使用空格分隔

> 删除全部：`docker rmi $(docker images -qa)`

**`OPTIONS`：**

- `-f`：强制删除

### 容器命令

#### `docker run`

启动容器

**`OPTIONS`：**

- `--name`：为容器指定一个名字
- `-d`：后台运行容器，并返回容器 `ID`，即启动守护容器
- `-i`：以交互式模式运行容器，通常与 `-t` 同时使用：`-it`
- `-t`：为容器重新分配一个伪输入终端
- `-P`：随机端口映射
- `-p`：指定端口映射

**端口映射格式：**

- `dockerPort:hostPort:containerPort`
- `dockerPort::containerPort`
- `hostPort:containerPort` **重要**
- `containerPort`

#### `docker ps`

列出所有正在运行的容器

**`OPTIONS`：**

- `-a`：列出当前所有正在运行的容器和历史上运行的容器
- `-l`：显示最近创建的容器 `l` 指 `last`
- `-n <num>`：显示最近 `n` 个创建的容器
- `-q`：静默模式，只显示容器编号
- `--no-trunc`：不截断输出

#### 退出容器

两种退出方式：

1. `exit`：容器停止退出
2. `ctrl + P + Q`：容器不停止退出

#### 进入容器

- `docker exec -it`：进入并执行，**然后再出来**
- `docker attach`：进入容器，等价于 `docker exec -it /bin/bash`

#### `docker stop`

停止容器

#### `docker start`

启动容器

#### `docker restart`

重启容器

#### `docker kill`

强制关闭容器

#### `docker rm`

删除已停止容器

> 区别于 `docker rmi`，`rmi` 是删除镜像，不是容器

**删除多个：**

`docker rm -f $(docker ps -qa)`

`docker ps -qa | xargs docker rm`

#### `docker logs`

查看容器日志

**`OPTIONS`：**

- `-f`：跟随最新的日志打印
- `-t`：加入时间戳
- `--tail <num>`：显示最后多少条日志

#### `docker top`

查看容器内进程

#### `docker inspect`

查看容器内部细节

#### `docker cp <id>:<path> <local path>`

将文件从容器拷贝到本地主机

## 四、`Docker` 镜像

### 简介

镜像是一种轻量级、可执行的独立软件包，用来打包软件运行环境和基于运行环境开发的软件，它包含运行某个软件所需的所有内容，包括代码、运行时、库、环境变量和配置文件

#### `UnionFS`

联合文件系统，是一种分层、轻量级并且高性能的文件系统，它支持对文件系统的修改作为一次提交来一层层的叠加，同时可以将不同目录挂载到同一个虚拟文件系统下，联合文件系统是 `Docker` 镜像的基础，镜像可以通过分层来进行继承，基于基础镜像（没有父镜像），可以制作各种具体的应用镜像。

**特性：**一次同时加载多个文件系统，但从外面来看，只能看到一个文件系统，联合加载会把各层文件系统叠加起来，这样最终的文件系统会包含所有底层的文件和目录

#### `Docker` 镜像加载原理

`bootfs(boot file system)`：主要包含 `bootloader` 和 `kernel`，`bootloader` 主要是引导加载 `kernel`，`Linux` 刚启动时会加载 `bootfs` 文件系统，在 `Docker` 镜像的最底层是 `bootfs`。这一层与典型的 `Linux/Unix` 系统是一样的，包含 `boot` 加载器和内核。当 `boot` 加载完成之后整个内核就都在内存中了，此时内存的使用权已由 `bootfs` 转交给内核，此时系统也会卸载 `bootfs`

`rootfs(root file system)`：在 `bootfs` 之上，包含的就是典型 `Linux` 系统中的 `/dev、/proc、/bin、/etc` 等标准目录和文件。`rootfs` 就是各种不同的操作系统发行版，比如 `Ubuntu`、`Centos` 等等

#### 镜像分层

镜像分层最大的好处就是资源共享

例如，有多个镜像都从相同的 `base` 镜像构建而来，那么宿主机只需要在磁盘上保存一份 `base` 镜像，同时内存中也只需加载一份 `base` 镜像，就可以为所有容器服务了，而且镜像的每一层都可以被共享

### 特点

1. `Docker` 镜像都是只读的
2. 当容器启动时，一个新的可写层被加载到镜像的顶部，**这一层通常被称作“容器层“，”容器层“之下的都叫做”镜像层“**

### 命令补充

#### `docker commit`

提交容器副本使之成为一个新的镜像

**`OPTIONS`：**

- `-m`：提交的描述信息
- `-a`：作者
- 源镜像 `ID`
- 要创建的目标镜像名:`tag`

## 五、`Docker` 容器数据卷

### 简介

`Docker` 将运用与运行的环境打包形成容器运行，运行可以伴随着容器，但是我们对数据的要求希望是持久化的

`Docker` 容器产生的数据，如果不通过 `docker commit` 生成新的镜像，使得数据做为镜像的一部分保存下来，那么当容器删除后，数据自然也就没有了。

为了能保存数据在 `docker` 中，我们使用卷

卷是目录或文件，存在于一个或多个容器中，由 `docker` 挂载到容器，但不属于联合文件系统，因此能够绕过联合文件系统提供一些用于持续存储或共享数据的特性

**卷的设计目的就是数据持久化**，**完全独立于容器的生存周期**，因此 `Docker` 不会在容器删除时删除其挂载的数据卷

**特点：**

1. 数据卷可在容器之间共享或重用数据
2. 卷中的更改可以直接生效
3. 数据卷中的更改不会包含在镜像的更新中
4. 数据卷的生命周期一直持续到没有容器使用它为止

### 添加数据卷

#### 直接命令添加

**命令：**`docker run it -v` /宿主机绝对路径目录:/容器内目录 镜像名

> 宿主机目录和容器内目录数据共享

**检查是否挂载成功：**

使用 `docker inspect`

**数据共享：**

1. 在主机中，往宿主机目录中添加文件 `-->` 在容器内目录可以查看到添加的文件
2. 在容器中，往容器目录中添加文件 `-->` 在主机中的目录中可以查看到添加的文件
3. 容器停止并退出后（不删除），在主机修改宿主目录中的内容 `-->`，容器重新启动后，会同步主机在其目录中的修改

> 类似于 `Windows` 的共享文件夹

**只读模式：**

如果 `docker` 容器内目录不允许写入，则需要使用如下命令：

`docker run it -v` /宿主机绝对路径目录:/容器内目录:`ro` 镜像名 

执行上述命令后，宿主机对共享目录的修改可以同步到容器，但是容器只能查看共享目录，而不能对目录内容进行修改

#### `DockerFile` 添加

**根目录下新建 `mydocker` 文件夹并进入**

**在 `DockerFile` 中使用 `VOLUME` 指令来给镜像添加一个或多个数据卷**

容器内数据卷：`VOLUME["/dataVolumeContainer", "/dataVolumeContainer2", "/dataVolumeContainer3"]`，在宿主机上，会有一个默认的对应目录，通过 `docker inspect` 查看

**构建 `File`**

**`build` 后生成镜像**

`docker build -f /mydocker/dockerfile2 -t `命名空间/镜像名 `output_dir`

**`run` 容器**

### 数据卷容器

#### 简介

命名的容器挂载数据卷，其他容器通过挂载这个（父容器）实现数据共享，挂载数据卷的容器，称之为数据卷容器

#### 容器间传递共享

**命令：**

**`docker run it root_doc --name doc_01`**

启动 `root_doc`，并重命名为 `doc_01`，`doc_01`拥有两个共享文件夹

**`docker run it root_doc --volumes-from doc_01 --name doc_02`**

启动第二个 `root_doc`，并重命名为 `doc_02`，不过这次启动的容器的共享文件夹连接的是 `doc_01` 而不是宿主机

**`docker run it root_doc --volumes-from doc_01 --name doc_03`**

启动第三个 `root_doc`，并重命名为 `doc_03`，这次启动的容器的共享文件夹同样连接到 `doc_01`

这样宿主机、`doc_01`、`doc_02`、`doc_03` 之间可以共享数据

**如果现在删除 `doc_01`，`doc_02` 和 `doc_03` 之间还是可以共享数据，并可以和宿主机共享**

**容器直接配置信息的传递，数据卷的生命周期一直持续到没有容器使用它为止**

## 六、`DockerFile` 解析

### 简介

`DockerFile` 是用来构建 `Docker` 镜像的构建文件，是由一系列命令和参数构成的脚本

**构建三步骤：**

1. 编写 `DockerFile` 文件
2. `docker build`
3. `docker run`

### `DockerFile` 构建过程解析

#### `DockerFile` 内容基础知识

1. 每条保留字指令都必须为大写字母且后面要跟随至少一个参数
2. 指令按照从上到下，顺序执行
3. `#` 表示注释
4. 每条指令都会创建一个新的镜像层，并对镜像进行提交

#### 执行流程

1. `docker` 从基础镜像运行一个容器
2. 执行一条指令并对容器作出修改
3. 执行类似 `docker commit` 的操作提交一个新的镜像层
4. `docker` 再基于刚提交的镜像运行一个新容器
5. 执行 `DockerFile` 中的下一条指令直到所有指令都执行完成

#### 总结

从应用软件的角度来看，`DockerFile`、`Docker` 镜像和 `Docker` 容器分别代表软件的三个不同阶段：

- `DockerFile` 是软件的原材料
- `Docker` 镜像是软件的交付品
- `Docker` 容器则可以认为是软件的运行态

`DockerFile` 面向开发，`Docker` 镜像称为交付标准，`Docker` 容器则涉及部署与运维，三者缺一不可，合力充当 `Docker` 体系的基石

- `DockerFile` 定义了进程需要的一切东西，涉及的内容包括执行代码或者是文件、环境变量、依赖包、运行时环境、动态链接库、操作系统的发行版、服务进程和内核进程（当应用进程需要和系统服务和内核进程打交道，这时候需要考虑如何设计 `namespace` 的权限控制）等等
- `Docker` 镜像，在用 `DockerFile` 定义一个文件之后，`docker build` 时会产生一个 `Docker` 镜像，当运行 `Docker` 镜像时，会真正开始提供服务
- `Docker` 容器，容器直接提供服务

### `DockerFile` 指令

#### `FROM`

基础镜像，当前新镜像是基于哪个镜像的

#### `MAINTAINER`

镜像维护者的名字和邮箱地址

#### `RUN`

容器构建时需要运行的命令

#### `EXPOSE`

当前容器对外暴露的端口

#### `WORKDIR`

指定在创建容器后，终端默认登录进来的工作目录

#### `ENV`

用来在构建镜像过程中设置环境变量

#### `ADD`

在宿主机目录下的文件拷贝进镜像，另外 `ADD` 命令会自动处理 `URL` 和解压 `tar` 压缩包

#### `COPY`

类似 `ADD`，拷贝文件或目录到镜像中

将从构建上下文目录中 <源路径> 的文件/目录复制到新的一层的镜像内的 <目标路径> 位置

#### `VOLUME`

容器数据卷，用于数据保存和持久化工作

#### `CMD`

指定一个容器启动时要运行的命令

`DockerFile` 中可以有多个 `CMD` 指令，但只有最后一个生效，`CMD` 会被 `docker run` 之后的参数替换

#### `ENTRYPOINT`

指定一个容器启动时要执行的命令

`ENTRYPOINT` 和 `CMD` 的目的一样，都是在指定容器启动程序及参数，`ENTRYPOINT` 不会覆盖之前的命令，会追加

#### `ONBUILD`

当构建一个被继承的 `DockerFile` 时运行命令，父镜像在被子镜像继承后父镜像的 `onbuild` 被触发

### 案例

#### `Base` 镜像

`FROM scratch`

#### 自定义镜像 `mycentos`

```dockerfile
from centos

ENV mypath /tmp
WORKDIR $mypath

RUN yum -y install vim
RUN yum -y install net-tools

EXPOSE 80
CMD /bin/bash
```

`docker build -f /mydocker/DockerFile -t mycentos:1.3 .`

`docker run -it mycentos:1.3`

#### `CMD/ENTRYPOINT` 镜像案例

无

#### 自定义镜像 `Tomcat9`

```dockerfile
FROM centos
MAINTAINER Guo Changyu<1028677200@qq.com>

COPY c.txt /usr/local/cincontainer.txt

ADD jdk-8u171-linux-x64.tar.gz /use/local
ADD apache-tomcat-9.0.8.tar.gz /usr/local

RUN yum -y install vim

ENV MYPATH /usr/local
WORKDIR $MYPATH

ENV JAVA_HOME /usr/local/jdk1.8.0_171
ENV CLASSPATH $JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
ENV CATALINA_HOME /usr/local/apache-tomcat-9.0.8
ENV CATALINA_BASE /usr/local/apache-tomcat-9.0.8
ENV PATH $PATH:$JAVA_HOME/bin:$CATALINA_HOME/lib:$CATALINA_HOME/bin

EXPOSE 8080

```



## 七、`Docker` 常用安装

## 八、本地镜像发布到阿里云
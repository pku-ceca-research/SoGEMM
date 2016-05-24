# SoGEMM
GEMM implementation on SDSoC

## 编译方式

首先请确定SDSoC的开发环境是否配置完成。具体配置方式：

```
source /where/SDSoC/is/located/settings64.sh
```

之后，在本目录下新建一个目录（比如命名为`build`），移动到`build/`目录下，进行如下操作：
```
ln -s ../Makefile.template Makefile
cp ../Makefile.config.example Makefile.config
```

`Makefile.config`是编译配置文件，其中设置的变量在`Makefile`中都有默认值，如果要进行修改请直接将`Makefile.config`中对应变量的注释去掉。

几个重要的设置：

1. `SDS`：编译环境是否为SDSoC工具链，如果是则设置为1，否则使用gcc/g++
2. `HW`：是否编译硬件代码，如果是则设置为1，SDSoC会进而将对应的函数作为目标进行HLS

配置好之后直接`make`即可，会生成一个包含`libaccel-[PLATFORM]-[sw/hw].so`的`sd_card`目录。

FROM nvcr.io/nvidia/pytorch:24.05-py3

# Add Mellanox GPG key to the new keyring directory
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox -O /etc/apt/trusted.gpg.d/mellanox.asc

# Add Mellanox repository
RUN cd /etc/apt/sources.list.d/ && wget https://linux.mellanox.com/public/repo/mlnx_ofed/latest/ubuntu22.04/mellanox_mlnx_ofed.list

# Update package list and upgrade existing packages
RUN apt-get -qq update && apt-get -qq upgrade -y

#instalingl all Mellanox OFED packages
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
    ibverbs-utils libibverbs-dev libibumad3 libibumad-dev librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils || apt-get -f install -y 

# this fixes IB errors that pops up
RUN rm -rf /etc/libibverbs.d/*.driver

# deleting nvidia examples for space
RUN rm -rf /workspace

#adding ping for network debug and ssh for openmpi
RUN apt update
RUN apt install inetutils-ping net-tools iproute2 openssh-server sysstat dnsutils -y --fix-missing

RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config && \
    mkdir /var/run/sshd -p
RUN pip install trl peft datasets transformers resnest

#creating my workspace
WORKDIR /workspace/
COPY training /workspace/training
WORKDIR /workspace/training
RUN chmod +x /workspace/training/entrypoint.sh
CMD ["sleep", "86400"]

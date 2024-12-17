import argparse
import os
import yaml
import json

def delete_yaml_files():
    for filename in os.listdir():
        if filename.endswith(".yaml"):
            os.remove(filename)

def generate_network_attachment_yaml(port, ip):
    config = {
        "cniVersion": "0.3.1",
        "name": f"network-port-{port}",
        "type": "sriov",
        "vlan": 0,
        "vlanQoS": 0,
        "logLevel": "info",
        "ipam": {
            "type": "whereabouts",
            "range": f"192.168.1.{ip}/24",
            "exclude": [
                "192.168.1.1",
                "192.168.1.2",
                "192.168.1.254",
                "192.168.1.255",
            ],
            "routes": [
                {
                    "dst": "192.168.1.0/24"
                }
            ]
        }
    }

    yaml_data = {
        "apiVersion": "k8s.cni.cncf.io/v1",
        "kind": "NetworkAttachmentDefinition",
        "metadata": {
            "annotations": {
                "k8s.v1.cni.cncf.io/resourceName": f"openshift.io/port{port}",
            },
            "name": f"network-port-{port}",
            "namespace": "default"
        },
        "spec": {
            "config": json.dumps(config, indent=4, separators=(',', ': '))
        }
    }
    return yaml_data

def generate_sriov_network_policy_yaml(port, num_vfs):
    yaml_data = {
        "apiVersion": "sriovnetwork.openshift.io/v1",
        "kind": "SriovNetworkNodePolicy",
        "metadata": {
            "name": f"mlnx-port-{port}",
            "namespace": "openshift-sriov-network-operator"
        },
        "spec": {
            "nodeSelector": {
                "feature.node.kubernetes.io/network-sriov.capable": "true"
            },
            "nicSelector": {
                "vendor": "15b3",
                "pfNames": ["ens3f0np0", "ens3f1np1"]
            },
            "deviceType": "netdevice",
            "numVfs": num_vfs,
            "priority": 99,
            "resourceName": f"port{port}",
            "isRdma": True
        }
    }
    return yaml_data

def main():
    parser = argparse.ArgumentParser(description="Generate YAML files with incremented port and IP address values.")
    parser.add_argument("--num-files", type=int, default=3, help="Number of files to generate (default: 3)")
    parser.add_argument("--starting-port", type=int, default=1, help="Starting port number (default: 1)")
    parser.add_argument("--starting-ip", type=int, default=2, help="Starting IP address (default: 2)")
    parser.add_argument("--num-vfs", type=int, default=1, help="Number of VFs for SriovNetworkNodePolicy (default: 1)")
    args = parser.parse_args()

    delete_yaml_files()  # Delete existing YAML files

    for i in range(args.num_files):
        port = args.starting_port + i
        ip = args.starting_ip + i

        network_attachment_yaml = generate_network_attachment_yaml(port, ip)
        sriov_network_policy_yaml = generate_sriov_network_policy_yaml(port, args.num_vfs)

        with open(f'network_attachment_{port}.yaml', 'w') as file:
            yaml.dump(network_attachment_yaml, file, default_flow_style=False)

        with open(f'sriov_network_policy_{port}.yaml', 'w') as file:
            yaml.dump(sriov_network_policy_yaml, file, default_flow_style=False)

if __name__ == "__main__":
    main()

/*
Copyright © 2019 Itay Shakury @itaysk

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package cmd

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	_ "strings"
	"unicode"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"
)

var outputFormat *string
var inputFile *string

func init() {
	outputFormat = rootCmd.PersistentFlags().StringP("output", "o", "yaml", "output format: yaml or json")
	inputFile = rootCmd.Flags().StringP("file", "f", "-", "file path to neat, or - to read from stdin")
	rootCmd.Flags().IntP("level", "l", 4, "neat level (0-4), the higher the neater")
	rootCmd.SetOut(os.Stdout)
	rootCmd.SetErr(os.Stderr)
	rootCmd.MarkFlagFilename("file")
	rootCmd.AddCommand(getCmd)
	getCmd.Flags().IntP("level", "l", 4, "neat level (0-4), the higher the neater")
}

// Execute is the entry point for the command package
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		rootCmd.PrintErr(err) // can't use PrintErrln or PrintErrf due to a bug https://github.com/spf13/cobra/pull/894
		os.Exit(1)
	}
}

var rootCmd = &cobra.Command{
	Use: "kubectl-neat",
	Example: `kubectl get pod mypod -o yaml | kubectl neat
kubectl get pod mypod -oyaml | kubectl neat -o json
kubectl neat -f - <./my-pod.json
kubectl neat -f ./my-pod.json
kubectl neat -f ./my-pod.json --output yaml`,
	RunE: func(cmd *cobra.Command, args []string) error {
		var in, out []byte
		var err error
		if *inputFile == "-" {
			stdin := cmd.InOrStdin()
			in, err = ioutil.ReadAll(stdin)
		} else {
			in, err = ioutil.ReadFile(*inputFile)
			if err != nil {
				return err
			}
		}
		outFormat := *outputFormat
		if !cmd.Flag("output").Changed {
			outFormat = "same"
		}

		level, _ := cmd.Flags().GetInt("level")
		out, err = NeatYAMLOrJSON(in, outFormat, level)
		if err != nil {
			return err
		}
		// XXX Hack: filter out version and generation
		lines := strings.Split(string(out), "\n")
		var newLines []string
		for _, line := range lines {
			//if strings.Contains(line, "version:") || strings.Contains(line, "generation:") {
			if strings.Contains(line, "generation:") {
				continue
			}
			newLines = append(newLines, line)
		}
		output := strings.Join(newLines, "\n")
		cmd.Print(output)

		//cmd.Print(string(out))
		return nil
	},
}

var kubectl = "kubectl"

var getCmd = &cobra.Command{
	Use: "get",
	Example: `kubectl neat get -- pod mypod -oyaml
kubectl neat get -- svc -n default myservice --output json`,
	FParseErrWhitelist: cobra.FParseErrWhitelist{UnknownFlags: true}, //don't try to validate kubectl get's flags
	RunE: func(cmd *cobra.Command, args []string) error {
		var out []byte
		var err error
		//reset defaults
		//there are two output settings in this subcommand: kubectl get's and kubectl-neat's
		//any combination of those can be provided by using the output flag in either side of the --
		//the most efficient is kubectl: json, kubectl-neat: yaml
		//0--0->Y--J #choose what's best for us
		//0--Y->Y--Y #user did specify output in kubectl, so respect that
		//0--J->J--J #user did specify output in kubectl, so respect that
		//Y--0->Y--J #user doesn't care about kubectl so use json but convert back
		//J--0->J--J #user expects json so use it for foth
		//if the user specified both side we can't touch it

		//the desired kubectl get output is always json, unless it was explicitly set by the user to yaml in which case the arg is overriden when concatenating the args later
		cmdArgs := append([]string{"get", "-o", "json"}, args...)
		kubectlCmd := exec.Command(kubectl, cmdArgs...)
		kres, err := kubectlCmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("Error invoking kubectl as %v %v", kubectlCmd.Args, err)
		}
		//handle the case of 0--J->J--J
		outFormat := *outputFormat
		kubeout := "yaml"
		for _, arg := range args {
			if arg == "json" || arg == "ojson" {
				outFormat = "json"
			}
		}
		if !cmd.Flag("output").Changed && kubeout == "json" {
			outFormat = "json"
		}
		level, _ := cmd.Flags().GetInt("level")
		out, err = NeatYAMLOrJSON(kres, outFormat, level)
		if err != nil {
			return err
		}
		cmd.Println(string(out))
		return nil
	},
}

func isJSON(s []byte) bool {
	return bytes.HasPrefix(bytes.TrimLeftFunc(s, unicode.IsSpace), []byte{'{'})
}

// NeatYAMLOrJSON converts 'in' to json if needed, invokes neat, and converts back if needed according the the outputFormat argument: yaml/json/same
func NeatYAMLOrJSON(in []byte, outputFormat string, level int) (out []byte, err error) {
	var injson, outjson string
	itsYaml := !isJSON(in)
	if itsYaml {
		injsonbytes, err := yaml.YAMLToJSON(in)
		if err != nil {
			return nil, fmt.Errorf("error converting from yaml to json : %v", err)
		}
		injson = string(injsonbytes)
	} else {
		injson = string(in)
	}

	outjson, err = Neat(injson, level)
	if err != nil {
		return nil, fmt.Errorf("error neating : %v", err)
	}

	if outputFormat == "yaml" || (outputFormat == "same" && itsYaml) {
		out, err = yaml.JSONToYAML([]byte(outjson))
		if err != nil {
			return nil, fmt.Errorf("error converting from json to yaml : %v", err)
		}
	} else {
		out = []byte(outjson)
	}
	return
}

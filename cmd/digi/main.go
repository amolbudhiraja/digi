package main

import (
	"fmt"
	"os"

	"digi.dev/digi/cmd/digi/digi"
)

func main() {
	if err := digi.RootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

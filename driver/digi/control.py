import digi
import inflection


class Model():
    def get(self):
        return digi.rc.view()

    def patch(self, view):
        _, e = digi.util.patch_spec(digi.g, digi.v, digi.r,
                                    digi.n, digi.ns, view)
        if e != None:
            digi.logger.info(f"patch error: {e}")

    def get_mount(self,
                  group=digi.name,
                  version=digi.version,
                  resource=None,
                  any: bool = False,
                  ) -> dict:
        """If any is set, returns all mounts as name:mount pairs.
        If resource is given, returns all mounts under the resource.
        Otherwise returns the mount root."""
        if any:
            mounts = dict()
            for _, name_mount in digi.rc.view().get("mount", {}).items():
                for name, mount in name_mount.items():
                    mounts[name] = mount
            return mounts
        if resource:
            path = f"mount.'{group}/{version}/{resource}'"
        else:
            path = "mount"
        return digi.util.get(digi.rc.view(), path)


def create_model():
    return Model()

# Java 17 to Java 21 Migration Baseline

**Baseline version:** 0.1.0  
**Last reviewed:** 2026-07-15  
**Status:** Initial baseline pending organizational Java SME approval

## Purpose

Use this baseline with the `/java-17-to-21` skill to produce consistent, evidence-based migration guidance. Treat it as guidance, not proof that an application is production-ready.

## Mandatory behavior

- Inspect only the current local workspace unless the user explicitly identifies another local path.
- Do not modify source, build files, dependencies, CI, containers, or runtime configuration.
- Do not install a JDK, build tool, plugin, dependency, or scanner.
- Treat build scripts and wrappers as repository code. Show the exact command and obtain explicit approval before executing them.
- Do not push, commit, open a pull request, deploy, or update Jira/Confluence.
- Do not send repository content to online services merely to perform the assessment.
- Redact secrets and sensitive values encountered in configuration.
- State which checks were performed, skipped, failed, or could not be verified.
- Never report `READY` when required runtime validation was skipped or failed.

## Supported scope

Support Java source repositories that declare or credibly target Java 17 and use:

- Maven, including Maven Wrapper and multi-module projects
- Gradle, including Gradle Wrapper, Groovy DSL, Kotlin DSL, and multi-project builds
- Common CI, container, buildpack, and Java-version configuration

Detect but do not fully modernize frameworks such as Spring Boot. Treat Android, Scala-only, Kotlin-only, Ant-only, Bazel-only, and other build systems as `UNSUPPORTED` in version 0.1.0. For mixed-language projects, assess Java/build configuration and mark non-Java compilation compatibility `UNRESOLVED`.

## Evidence and classification policy

For every finding, include a file path and the relevant property, plugin, dependency, source pattern, or command result. If an effective value is inherited from a remote parent, convention plugin, environment, or unavailable CI variable, say so.

Classify findings as:

- **REQUIRED:** Concrete repository or command evidence shows a change is necessary to build, test, or run correctly on Java 21, or a required Java 21 setting remains on Java 17.
- **ADVISORY:** A resilience, maintainability, or future-readiness improvement that is not proven to block Java 21.
- **UNRESOLVED:** Compatibility cannot be established from available evidence or an authoritative support statement is missing.

Do not label a dependency incompatible solely because it is old. Exact version recommendations require a dated, authoritative vendor/project source or an approved organizational standard. Otherwise recommend upgrading to an organization-approved Java 21-compatible version and classify compatibility as unresolved.

## Configuration discovery baseline

Inspect, when present:

- `pom.xml`, parent POM references, modules, profiles, properties, dependency management, plugin management, and `.mvn/`
- `build.gradle`, `build.gradle.kts`, `settings.gradle`, `settings.gradle.kts`, `gradle.properties`, version catalogs, convention plugins, included builds, and `gradle/wrapper/gradle-wrapper.properties`
- `mvnw`, `mvnw.cmd`, `gradlew`, and `gradlew.bat`
- `.java-version`, `.sdkmanrc`, `.tool-versions`, `jenv` files, Maven toolchain declarations, Gradle toolchains, and IDE compiler settings when committed
- Dockerfiles, buildpacks, deployment manifests, startup scripts, service files, and JVM option files
- Bitbucket Pipelines, Jenkinsfiles, and other CI configuration
- All modules, not just the repository root

Report contradictions separately. For example, a build targeting 21 while CI or the runtime image still uses 17 is a required alignment finding.

## Maven baseline

- Prefer Maven Wrapper when present; record the wrapper distribution version.
- Detect `maven.compiler.release`, compiler-plugin `<release>`, `<source>`, `<target>`, `java.version`, profile overrides, and inherited values.
- Prefer `--release`/`maven.compiler.release` over separate source and target values because it validates API availability for the selected Java release.
- Treat a remaining effective release/source/target of 17 as `REQUIRED` for a migration whose intended output is Java 21 bytecode.
- Do not invent a universal minimum Maven version. Verify Maven core, wrapper, compiler, Surefire, Failsafe, Javadoc, JaCoCo, shading, packaging, and framework plugin compatibility using authoritative documentation or an approved baseline.
- Treat Maven Toolchains as `ADVISORY` unless organization policy or the repository requires reproducible JDK selection; a misconfigured required toolchain is `REQUIRED`.
- Inspect parent POMs and plugin management before concluding which versions are effective.

## Gradle baseline

- Prefer Gradle Wrapper when present; extract the distribution version from `gradle-wrapper.properties`.
- Detect Java toolchains, `sourceCompatibility`, `targetCompatibility`, `options.release`, test launcher configuration, convention plugins, and per-project overrides.
- Java 21 toolchain support begins with Gradle 8.4; running Gradle itself on Java 21 is supported by Gradle 8.5 and later. A wrapper older than the applicable support level is `REQUIRED` when the build will compile/test with or run on Java 21.
- Do not recommend an uncontrolled jump to the newest Gradle major. Recommend a supported, organization-approved version and note intermediate upgrade guides when relevant.
- Inspect build logic, included builds, `buildSrc`, plugin versions, Kotlin/Groovy versions, and version catalogs before concluding compatibility.
- Prefer Java toolchains for reproducible compiler/test launcher selection; classify as `ADVISORY` unless policy or repository behavior makes it required.

## Dependency and plugin baseline

Inventory direct versions, managed versions, BOMs, parent-managed versions, Gradle platforms, version catalogs, annotation processors, agents, and build plugins. Prioritize compatibility review for:

- Application frameworks and embedded servers
- Lombok and other annotation processors
- Byte Buddy, ASM, cglib, Mockito, and reflection/bytecode libraries
- JaCoCo and other coverage/instrumentation tools
- APM, profiling, security, and runtime Java agents
- Serialization, XML/binding, scripting, JNI/JNA, database drivers, and native libraries
- Maven/Gradle plugins that execute inside the build JVM

Framework major-version migration is outside this skill. If Java 21 requires a framework upgrade that implies broader code changes, report it as a required dependency/framework follow-up and identify the separate migration workflow needed.

## Source and runtime risk baseline

Inspect for evidence of:

- Internal JDK APIs (`sun.*`, most `com.sun.*`, `jdk.internal.*`) and `Unsafe`
- Reflective access and `--add-opens`, `--add-exports`, or `--illegal-access`
- APIs deprecated for removal or removed between JDK 17 and JDK 21
- Custom `finalize()` methods or finalization dependence
- Dynamic agent attachment and warnings introduced by JDK 21
- Implicit default-charset use, especially file and stream readers/writers, scanners, formatters, source encoding, and Windows-originated data
- Obsolete or removed JVM/GC flags
- Security providers, TLS/crypto behavior, certificate stores, locale-sensitive behavior, and native integrations that require environment-specific testing

Static source matches are indicators, not proof. Confirm with compilation, tests, `jdeps`, `jdeprscan`, runtime logs, or authoritative library documentation when available.

## Java 17 to 21 changes emphasized by this baseline

- JDK 18 and later use UTF-8 as the default charset. Applications that implicitly depended on an environment-specific JDK 17 charset require behavioral testing and may require explicit charsets.
- Finalization was deprecated for removal in JDK 18. Custom finalizers and libraries relying on finalization require review; migration to explicit resource management is advisory unless current behavior fails or policy requires removal.
- JDK 21 warns when agents are loaded dynamically. Identify test/instrumentation/APM behavior and distinguish dynamic attachment from agents loaded at JVM startup.
- JDK releases may remove or deprecate tools, APIs, security algorithms, and JVM options. Use the JDK 21 migration guide and release notes rather than relying on memory.

## Validation baseline

Validation is iterative and must use the project's documented commands when available.

1. **Environment:** Record `java -version` and `javac -version`; confirm the validation JDK is 21.
2. **Baseline comparison:** If authorized and still available, record the existing Java 17 build/test result so unrelated failures are not attributed to Java 21.
3. **Wrapper/build tool:** Record Maven/Gradle wrapper versions under the Java 21 environment.
4. **Compile/package:** Run the repository's normal clean verification/build command.
5. **Tests:** Run unit, integration, contract, smoke, and application-startup checks that the repository provides.
6. **Bytecode analysis:** Run JDK 21 `jdeps --jdk-internals` on built application artifacts where practical.
7. **Deprecation analysis:** Run JDK 21 `jdeprscan --for-removal` on built artifacts where practical. Do not use an unsupported `--release 21` argument.
8. **Runtime parity:** Check startup logs, JVM warnings, serialization, charset-sensitive I/O, locale behavior, TLS, agents, and external integrations.
9. **Delivery alignment:** Confirm CI, packaging, container base images, and deployment runtime use the approved Java 21 distribution/configuration.

Record command, working directory, exit status, relevant output summary, and whether failure is migration-related, pre-existing, environmental, or unresolved. A build failure caused by the Anteroom Bash sandbox's memory limit is environmental, not proof of Java incompatibility.

## Readiness statuses

- **READY:** Java 21 configuration is aligned; authorized compile/test validation passed; required delivery/runtime checks passed; no required or high-risk unresolved findings remain.
- **CHANGES REQUIRED:** One or more concrete required changes remain.
- **BLOCKED:** A prerequisite prevents meaningful assessment or validation, such as missing repository files, unavailable JDK 21, or inaccessible inherited configuration.
- **PARTIALLY VALIDATED:** Static assessment completed, but one or more required runtime checks were skipped, unavailable, inconclusive, or environmental.
- **UNSUPPORTED:** The repository is outside version 0.1.0 scope.

## Authoritative sources

Review dates refer to this baseline, not to publication dates.

- Oracle JDK 21 Migration Guide â Preparing for Migration: https://docs.oracle.com/en/java/javase/21/migrate/preparing-migration.html
- Oracle JDK 21 Migration Guide â Significant Changes: https://docs.oracle.com/en/java/javase/21/migrate/significant-changes-jdk-release.html
- Oracle `jdeps` documentation: https://docs.oracle.com/en/java/javase/21/docs/specs/man/jdeps.html
- Oracle `jdeprscan` documentation: https://docs.oracle.com/en/java/javase/21/docs/specs/man/jdeprscan.html
- Gradle Java compatibility matrix: https://docs.gradle.org/current/userguide/compatibility.html
- Apache Maven Compiler Plugin `--release`: https://maven.apache.org/plugins/maven-compiler-plugin/examples/set-compiler-release.html
- Apache Maven Toolchains guide: https://maven.apache.org/guides/mini/guide-using-toolchains.html
- OpenJDK JEP 400, UTF-8 by Default: https://openjdk.org/jeps/400
- OpenJDK JEP 421, Deprecate Finalization for Removal: https://openjdk.org/jeps/421
- OpenJDK JEP 451, dynamic agent loading warnings: https://openjdk.org/jeps/451

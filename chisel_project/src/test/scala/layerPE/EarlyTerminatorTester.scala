package YoseUe_SATL

import chisel3._
import chiseltest._
import org.scalatest.freespec.AnyFreeSpec
import chisel3.experimental.BundleLiterals._
import scala.math._
import java.io.PrintWriter
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import dataclass.data
import java.nio.charset.StandardCharsets
import scala.io.Source
import chisel3.experimental.FixedPoint
import spatial_templates.pe._
import spatial_templates.me._

class EarlyTerminatorPETester extends AnyFreeSpec with ChiselScalatestTester {

  val n_attr = 5
  val n_classes = 6
  val n_depths = 5
  val info_bit = 10
  val tree_bit = 8

  "Pe should check whether the score of the sample is already decided" in {
    test(new EarlyTerminatorPE(new ElemId(2,0,0,0),n_attr,n_classes,n_depths,info_bit,tree_bit,180.0,2)) { c =>
        
        for (k <- 0 until 3){
            for (i <- 0 until 2){
                c.io.samples_in(i).bits.offset.poke(0.U)
                c.io.samples_in(i).bits.shift.poke(false.B)
                c.io.samples_in(i).bits.tree_to_exec.poke(1.U)
                c.io.samples_in(i).bits.dest.poke(false.B)
                for (j <- 0 until n_classes){
                    c.io.samples_in(i).bits.scores(j).poke(((13 - 3*j+k)).F(16.W,8.BP))
                }
                for (j <- 0 until n_depths){
                    c.io.samples_in(i).bits.weights(j).poke(i.F(16.W,8.BP))
                }
                for (j <- 0 until n_attr){
                    c.io.samples_in(i).bits.features(j).poke(5.F(32.W,16.BP))
                }
                if(i==0){
                    c.io.samples_in(i).valid.poke(true.B)
                }else{
                    c.io.samples_in(i).valid.poke(true.B)
                }
                //c.io.samples_in(i).valid.poke(true.B)
                c.io.sample_out.ready.poke(true)
            }

            c.clock.step(1)
        }
        
        c.clock.step(1)
        /*
        println("SCORES: ")
        for (i <- 0 until n_classes){
            val score = c.io.sample_out.bits.scores(i).peek().litValue
            val fixedPointValue: Double = (score >> 8).toDouble + ((score & BigInt("FF", 16)).toDouble) / pow(2, 8)
            println(fixedPointValue)
        }
        println("VALID: ")
        println(c.io.sample_out.valid.peek().litValue)

        println("VALID BACK: ")
        println(c.io.samples_back(0).valid.peek().litValue)
        */
        println("Emitting verilog")
        val VerilogEmitter = (new chisel3.stage.ChiselStage).emitVerilog(
            new EarlyTerminatorPE(new ElemId(2,0,0,0),n_attr,n_classes,n_depths,info_bit,tree_bit,180.0,2)
        )
        Files.write(
            Paths.get("./EarlyTerminatorWithSort.v"),
            VerilogEmitter.getBytes(StandardCharsets.UTF_8)
        )
        
        println("Verilog emitted")
        
    }
  }
}